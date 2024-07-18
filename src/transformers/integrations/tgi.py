import importlib.metadata
import warnings
from copy import deepcopy
from inspect import signature

from packaging import version

from ..utils import is_accelerate_available, is_bitsandbytes_available, logging
from glm_kernel import (
    weight_only_quant_ops,
    fused_gemm_ops
)
from typing import Tuple

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    from ..pytorch_utils import Conv1D

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters

logger = logging.get_logger(__name__)


def tgi_quantize(weight :torch.Tensor, weight_bit_width=8,
                 fp8_activation=False, group_size=None, has_zeros=False) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
    a_dtype = 0 if weight.dtype == torch.float16 else 1
    if fp8_activation:
        a_dtype = 2
    # padding weight last dim to multiple of 64 and group_size
    if weight.shape[-1] % 64 != 0:
        weight = torch.nn.functional.pad(weight, (0, 64 - weight.shape[-1] % 64))
    if group_size is not None and weight.shape[-1] % group_size != 0:
        weight = torch.nn.functional.pad(weight, (0, group_size - weight.shape[-1] % group_size))

    quantizated_weight = weight.cpu().to(torch.float32)
    quantizated_weight = quantizated_weight.view(weight.shape[0], -1, weight.shape[-1] if group_size is None else group_size)

    if not has_zeros:
        weight_scale = quantizated_weight.abs().max(dim=-1).values / (2 ** (weight_bit_width - 1) - 1)
        quantizated_weight = torch.round(quantizated_weight / weight_scale.unsqueeze(-1)).to(torch.int8).view(quantizated_weight.shape[0], -1).t().contiguous().cpu()
    else:
        weight_zero = quantizated_weight.min(dim=-1).values
        quantizated_weight = quantizated_weight - weight_zero.unsqueeze(-1)
        weight_scale = quantizated_weight.max(dim=-1).values / (2 ** weight_bit_width - 1)
        quantizated_weight = torch.round(quantizated_weight / weight_scale.unsqueeze(-1) - 2 ** (weight_bit_width - 1)).to(torch.int8)
        weight_zero = weight_zero + 2 ** (weight_bit_width - 1) * weight_scale
        quantizated_weight = quantizated_weight.view(quantizated_weight.shape[0], -1).t().contiguous().cpu()

    if weight_bit_width == 4:
        quantizated_weight = weight_only_quant_ops.pack_int8_tensor_to_packed_int4(quantizated_weight)
    quantizated_weight = weight_only_quant_ops.preprocess_weights_for_mixed_gemm(quantizated_weight, weight_bit_width, a_dtype)
    quantizated_weight = torch.nn.Parameter(quantizated_weight.to(weight.device), requires_grad=False)

    weight_scale = weight_scale.to(weight.dtype).t().contiguous()
    if group_size is None:
        weight_scale = weight_scale.squeeze(0).contiguous()
    weight_scale = torch.nn.Parameter(weight_scale.to(weight.device), requires_grad=False)

    if has_zeros:
        weight_zero = weight_zero.to(weight.dtype).t().contiguous()
        weight_zero = torch.nn.Parameter(weight_zero.to(weight.device), requires_grad=False)
    else:
        weight_zero = None

    torch.cuda.empty_cache()
    return quantizated_weight, weight_scale, weight_zero

class WeightOnlyQuantize(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_zero = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = torch.nn.Parameter(bias.to(weight.device, dtype=self.weight_scale.dtype), requires_grad=False) if bias is not None else None
        return

    def forward(self, x: torch.Tensor):
        x_shape = x.size()
        # padding x if x not the same shape as weight
        if x_shape[-1] != self.weight.size(0):
            x = torch.nn.functional.pad(x, (0, self.weight.size(0) - x_shape[-1]))
        x = x.contiguous().view(-1, x.size(-1))
        if self.weight_scale.dim() == 1:
            out_features = self.weight_scale.size(0)
            output = fused_gemm_ops.fused_gemm(x, self.weight, self.weight_scale, self.bias)
        elif self.weight_zero is None:
            out_features = self.weight_scale.size(1)
            group_size = self.weight.shape[0] // self.weight_scale.shape[0]
            output = fused_gemm_ops.fused_gemm_group(x, self.weight, self.weight_scale, group_size, self.bias)
        else:
            out_features = self.weight_scale.size(1)
            group_size = self.weight.shape[0] // self.weight_scale.shape[0]
            output = fused_gemm_ops.fused_gemm_group_zero(x, self.weight, self.weight_scale, self.weight_zero, group_size, self.bias)
        return output.view(*(x_shape[:-1] + (out_features,)))


def _replace_with_tgi_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    # model.to(torch.cuda.current_device())
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    if (
                        quantization_config.llm_int8_skip_modules is not None
                        and name in quantization_config.llm_int8_skip_modules
                    ):
                        pass
                    else:
                        model._modules[name] = WeightOnlyQuantize(
                            module.weight,
                            module.bias
                        )
                        has_been_replaced = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_tgi_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_tgi_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `LLM.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_tgi_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


# For backward compatibility
def replace_8bit_linear(*args, **kwargs):
    warnings.warn(
        "`replace_8bit_linear` will be deprecated in a future version, please use `replace_with_tgi_linear` instead",
        FutureWarning,
    )
    return replace_with_tgi_linear(*args, **kwargs)


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # If there is not tied weights, we want to keep the lm_head（output_embedding) in full precision
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # otherwise, no tied weights, no output embedding defined, simply keep the last module in full precision
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names

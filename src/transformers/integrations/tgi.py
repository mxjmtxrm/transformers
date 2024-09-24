import warnings
from copy import deepcopy
from inspect import signature
from ..utils import is_accelerate_available, is_tgi_available, logging
from glm_kernel import (
    weight_only_quant_ops
)
from typing import Tuple
import triton
import triton.language as tl
from itertools import product
import os
if is_tgi_available():
    import torch
    import torch.nn as nn


if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters

logger = logging.get_logger(__name__)

import os
if "TRITON_DEJAVU_STORAGE" not in os.environ:
    warmup_path = "/tmp/warmup"
    if not os.path.exists(warmup_path):
        os.mkdir(warmup_path)
    os.environ["TRITON_DEJAVU_STORAGE"] = "/tmp/warmup"

def glm_quantize(weight :torch.Tensor) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
    # padding weight last dim to multiple of 64 and group_size
    if weight.shape[-1] % 64 != 0:
        weight = torch.nn.functional.pad(weight, (0, 64 - weight.shape[-1] % 64)) # (n, k)
    weight_bit_width = 4
    quantizated_weight = weight.cpu().to(torch.float32)
    quantizated_weight = quantizated_weight.view(weight.shape[0], -1, weight.shape[-1])
    weight_scale, weight_zero = None, None
    weight_zero = quantizated_weight.min(dim=-1).values
    quantizated_weight = quantizated_weight - weight_zero.unsqueeze(-1)
    weight_scale = quantizated_weight.max(dim=-1).values / (2 ** weight_bit_width - 1)
    quantizated_weight = torch.round(quantizated_weight / weight_scale.unsqueeze(-1) - 2 ** (weight_bit_width - 1)).to(torch.int8)
    quantizated_weight = quantizated_weight.view(quantizated_weight.shape[0], -1).t().contiguous().cpu()  # (k, n)
    quantizated_weight = weight_only_quant_ops.pack_int8_tensor_to_packed_int4(quantizated_weight) # (k, n // 2)
    quantizated_weight = torch.nn.Parameter(quantizated_weight.to(weight.device), requires_grad=False)
    weight_scale = weight_scale.to(weight.dtype).t().contiguous()
    weight_scale = weight_scale.squeeze(0).unsqueeze(-1).contiguous()
    weight_scale = torch.nn.Parameter(weight_scale.to(weight.device), requires_grad=False)
    weight_zero = weight_zero.to(weight.dtype)
    weight_zero = torch.nn.Parameter(weight_zero.to(weight.device), requires_grad=False)

    torch.cuda.empty_cache()
    return quantizated_weight, weight_scale, weight_zero

# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import triton_dejavu
# To cache tune configs
@triton_dejavu.autotune(
    configs=[
         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=5, num_warps=4),
         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
     ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_zero_triton_kernel(output_ptr, output_row_stride, output_col_stride,
                                  input_ptr, input_row_stride, input_col_stride,
                                  weight_scale_ptr, weight_scale_stride,
                                  weight_zero_ptr, weight_zero_stride,
                                  exp_num:tl.constexpr,
                                  K: tl.constexpr, N: tl.constexpr, ori_shape:tl.constexpr,
                                  BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.max_contiguous(k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_n = tl.max_contiguous(n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_nw = tl.max_contiguous(n_block_idx * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2), BLOCK_SIZE_N // 2)
    offs_scale = offs_n * weight_scale_stride
    # offs_zero = offs_n * weight_zero_stride

    offs_output = offs_k[None, :] * output_col_stride + offs_n[:, None] * output_row_stride
    offs_weight = offs_k[:, None] * input_row_stride + offs_nw[None, :] * input_col_stride
    # offs_zero = offs_k
    # offs_scale = offs_k[:, None]
    # n_mask = offs_n[None, :] < N
    # k_mask = offs_k[:, None] < K
    n_mask = offs_n[:, None] < N
    k_mask = offs_k[None, :] < ori_shape
    mask = n_mask & k_mask
    weight_nmask = offs_nw[None, :] < (N // 2)
    weight_kmask = offs_k[:, None] < K
    weight_mask = weight_nmask & weight_kmask
    scale_mask = offs_scale < N
    # zero_mask = offs_zero < N

    int4_weight = tl.load(input_ptr + offs_weight, mask=weight_mask).to(tl.int8)
    row_0 = ((int4_weight << 4).to(tl.int8) >> 4).to(tl.int8)
    row_1 = (int4_weight >> 4).to(tl.int8)
    int8_weight = tl.join(row_0, row_1)
    int8_weight = int8_weight.reshape(row_0.shape[0], row_1.shape[1] * 2).permute(1, 0) # (k, n) -> (n, k)

    zero = tl.load(weight_zero_ptr + offs_scale, mask=scale_mask).to(tl.float16)
    zero = tl.expand_dims(zero, axis=-1)
    scale = tl.load(weight_scale_ptr + offs_scale, mask=scale_mask).to(tl.float16)
    scale = tl.expand_dims(scale, axis=-1)
    output = scale * (int8_weight + exp_num) + zero
    tl.store(output_ptr + offs_output, output, mask=mask)
    return


@triton_dejavu.autotune(
    configs=[
         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=5, num_warps=4),
         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
     ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_zero_triton_transposed_kernel(output_ptr, output_row_stride, output_col_stride,
                                  input_ptr, input_row_stride, input_col_stride,
                                  weight_scale_ptr, weight_scale_stride,
                                  weight_zero_ptr, weight_zero_stride,
                                  exp_num:tl.constexpr,
                                  K: tl.constexpr, N: tl.constexpr, ori_shape:tl.constexpr,
                                  BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.max_contiguous(k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_n = tl.max_contiguous(n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_nw = tl.max_contiguous(n_block_idx * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2), BLOCK_SIZE_N // 2)
    offs_scale = offs_n * weight_scale_stride

    offs_output = offs_k[:, None] * output_row_stride + offs_n[None, :] * output_col_stride
    offs_weight = offs_k[:, None] * input_row_stride + offs_nw[None, :] * input_col_stride
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < ori_shape
    mask = n_mask & k_mask
    weight_nmask = offs_nw[None, :] < (N // 2)
    weight_kmask = offs_k[:, None] < K
    weight_mask = weight_nmask & weight_kmask
    scale_mask = offs_scale < N

    int4_weight = tl.load(input_ptr + offs_weight, mask=weight_mask).to(tl.int8)
    row_0 = ((int4_weight << 4).to(tl.int8) >> 4).to(tl.int8)
    row_1 = (int4_weight >> 4).to(tl.int8)
    int8_weight = tl.join(row_0, row_1)
    int8_weight = int8_weight.reshape(row_0.shape[0], row_1.shape[1] * 2).permute(1, 0) # (k, n) -> (n, k)

    zero = tl.load(weight_zero_ptr + offs_scale, mask=scale_mask).to(tl.float16)
    zero = tl.expand_dims(zero, axis=-1)
    scale = tl.load(weight_scale_ptr + offs_scale, mask=scale_mask).to(tl.float16)
    scale = tl.expand_dims(scale, axis=-1)
    output = scale * (int8_weight + exp_num) + zero
    output = output.permute(1, 0)
    tl.store(output_ptr + offs_output, output, mask=mask)
    return

# m, n, k     weight: (k, n // 2), weight_scale: (n), weight_zero: (n), output: (n, k)
def dequantize_triton(weight, weight_scale, weight_zero, weight_bit_width, ori_shape, is_transposed=True):
    K, Nw = weight.shape
    N = Nw * 2
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    exp_num = 2 ** (weight_bit_width - 1)
    if is_transposed:
        output = torch.empty(size=(N, ori_shape), dtype=weight_scale.dtype, device=weight.device)
        assert weight.is_cuda and output.is_cuda
        dequantize_zero_triton_kernel[grid](
            output, output.stride(0), output.stride(1),
            weight, weight.stride(0), weight.stride(1),
            weight_scale, weight_scale.stride(0),
            weight_zero, weight_zero.stride(0),
            exp_num,
            K, N, ori_shape
        )
    else:
        output = torch.empty(size=(ori_shape, N), dtype=weight_scale.dtype, device=weight.device)
        assert weight.is_cuda and output.is_cuda
        dequantize_zero_triton_transposed_kernel[grid](
            output, output.stride(0), output.stride(1),
            weight, weight.stride(0), weight.stride(1),
            weight_scale, weight_scale.stride(0),
            weight_zero, weight_zero.stride(0),
            exp_num,
            K, N, ori_shape
        )
    return output

class GLMQuantize(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_bit_width=4
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.weight_scale = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_zero = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = torch.nn.Parameter(bias.to(weight.device, dtype=self.weight_scale.dtype), requires_grad=False) if bias is not None else None
        self.weight_bit_width = weight_bit_width
        self.ori_shape = weight.shape[-1]
        return

    def forward(self, x: torch.Tensor):
        output = GLMMatMulBit.apply(x, self.weight, self.weight_scale, 
                                    self.weight_zero, self.weight_bit_width, 
                                    self.ori_shape, self.bias)
        return output


class GLMMatMulBit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, weight_zero, weight_bit_width, ori_shape, bias):
        ctx.save_for_backward(weight, weight_scale, weight_zero, bias)
        ctx.dtype_bias = bias.dtype if bias is not None else None
        ctx.weight_bit_width = weight_bit_width
        ctx.ori_shape = ori_shape
        if bias is not None:
            return torch.matmul(x, dequantize_triton(weight, weight_scale, weight_zero, weight_bit_width, ori_shape, is_transposed=False)) + bias
        else:
            return torch.matmul(x, dequantize_triton(weight, weight_scale, weight_zero, weight_bit_width, ori_shape, is_transposed=False))

    @staticmethod
    def backward(ctx, grad_output):
        weight, weight_scale, weight_zero, bias = ctx.saved_tensors
        weight_bit_width = ctx.weight_bit_width
        ori_shape = ctx.ori_shape
        grad_output = grad_output
        req_gradX, _, _, _, _, _, req_gradBias = ctx.needs_input_grad
        grad_x, grad_bias = None, None
        if req_gradBias:
            grad_bias = grad_output.sum(0, dtype=bias.dtype)
        if req_gradX:
            grad_x = torch.matmul(grad_output, dequantize_triton(weight, weight_scale, weight_zero, weight_bit_width, ori_shape, is_transposed=True))
        return grad_x, None, None, None, None, None, grad_bias


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
    weight_bit_width = 4 if quantization_config.load_in_tgi_4bit else 8
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
                        model._modules[name] = GLMQuantize(
                            module.weight,
                            module.bias,
                            weight_bit_width = weight_bit_width,
                            # group_size=quantization_config.group_size
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

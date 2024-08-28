# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from packaging import version

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_torch_available, logging
from ..integrations import get_keys_to_not_convert, replace_with_tgi_linear, GLMQuantize, glm_quantize, dequantize_triton


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class TGIBitHfQuantizer(HfQuantizer):
    """
    4-bit quantization from bitsandbytes.py quantization method:
        before loading: converts transformer layers into Linear4bit during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear4bit into 4bit at the first .cuda() call
        saving:
            from state dict, as usual; saves weights and `quant_state` components
        loading:
            need to locate `quant_state` components and pass to Param4bit constructor
    """

    use_keep_in_fp32_modules = True
    requires_parameters_quantization = True
    requires_calibration = False
    warm_up_cache = set()
    # required_packages = ["bitsandbytes", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        device_map = kwargs.get("device_map", None)
        if (
            device_map is not None
            and isinstance(device_map, dict)
            and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
        ):
            device_map_without_lm_head = {
                key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
            }
            if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.39.0"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.19.0"):
            from accelerate.utils import CustomDtype

            if target_dtype != torch.int8:
                logger.info("target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization")
            return CustomDtype.INT4
        else:
            raise ValueError(
                "You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source to support fp4 auto device map"
                "calculation. You may encounter unexpected behavior, or pass your own device map"
            )
    
    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`List[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        tmp_missing_keys = missing_keys.copy()
        ignore_missing_keys = ["weight_scale", "weight_zero", "weight_bit_width", "ori_shape", "group_size"]
        for key in tmp_missing_keys:
            for ignore_missing_key in ignore_missing_keys:
                if ignore_missing_key in key:
                    missing_keys.remove(key)
        return missing_keys

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        module, _ = get_module_from_name(model, param_name)
        if isinstance(module, GLMQuantize):
            # Add here check for loaded components' dtypes once serialization is implemented
            return True
        return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        """
        remove unexpected_keys and set meta tensor to target device
        """

        module, tensor_name = get_module_from_name(model, param_name)
        if param_value is None:
            raise ValueError(f"tensor {param_name} is None")
        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
        # module._parameters[tensor_name].to(target_device)
        if tensor_name in ["weight_scale", "weight_zero"]:
            return
    
        if tensor_name == "bias":
            old_value = getattr(module, tensor_name)
            new_value = param_value.to(target_device).to(old_value.dtype)
            new_value = torch.nn.Parameter(new_value, requires_grad=False)
            module._parameters[tensor_name] = new_value
            return
        # weight_bit_width = 8
        # if self.quantization_config.load_in_tgi_4bit:
        #     weight_bit_width = 4
        n, k = param_value.shape
        weight, weight_scale, weight_zero = glm_quantize(param_value)
        module._parameters["weight"] = weight
        module._parameters["weight_scale"] = weight_scale
        module._parameters["weight_zero"] = weight_zero
        if "bias" in module._parameters:
            module._parameters["bias"] = module._parameters["bias"].to(weight_scale.dtype)

        # warm_up
        if torch.distributed.get_rank() == 0:
            if (n, k) in self.warm_up_cache:
                return
            self.warm_up_cache.add((n, k))
            weight_cuda = weight.to("cuda:0")
            weight_scale_cuda = weight_scale.to("cuda:0")
            weight_zero_cuda = weight_zero.to("cuda:0")
            weight_bit_width = 4
            dequantize_triton(weight_cuda, weight_scale_cuda, weight_zero_cuda, weight_bit_width, ori_shape=k, is_transposed=True)
            dequantize_triton(weight_cuda, weight_scale_cuda, weight_zero_cuda, weight_bit_width, ori_shape=k, is_transposed=False)
            torch.cuda.empty_cache()
        return

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.adjust_max_memory
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_torch_dtype
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_device_map
    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": torch.cuda.current_device()}
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':torch.cuda.current_device()}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        return device_map

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if self.quantization_config.llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_tgi_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

        # TODO: consider bringing replace_with_tgi_linear() code from ..integrations/bitsandbyter.py to here

        model.config.quantization_config = self.quantization_config

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_after_weight_loading with 8bit->4bit
    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable
        return model

    @property
    def is_serializable(self):
        _is_4bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.3")

        if not _is_4bit_serializable:
            logger.warning(
                "You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed."
            )
            return False

        return True

    @property
    def is_trainable(self) -> bool:
        return True

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    model_4bit = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", load_in_tgi_4bit=True)
    print(model_4bit)
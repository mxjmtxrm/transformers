# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import unittest

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TGIQuantizationConfig,
    pipeline,
)
import torch

def get_some_linear_layer(model):
    if model.config.model_type == "gpt2":
        return model.transformer.h[0].mlp.c_fc
    elif model.config.model_type == "opt":
        try:
            return model.decoder.layers[0].fc1
        except AttributeError:
            # for AutoModelforCausalLM
            return model.model.decoder.layers[0].fc1
    elif model.config.model_type == "chatglm":
        return model.transformer.encoder.layers[0].mlp.dense_4h_to_h
    else:
        return model.transformer.h[0].mlp.dense_4h_to_h

# @require_bitsandbytes
# @require_accelerate
# @require_torch
# @require_torch_gpu
# @slow
class Base4bitTest(unittest.TestCase):
    # We keep the constants inside the init function and model loading inside setUp function

    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    # model_name = "bigscience/bloom-1b7"
    model_name = "THUDM/glm-4-9b-chat"

    # Constant values
    EXPECTED_RELATIVE_DIFFERENCE = (
        2.658022431400294  # This was obtained on a NVIDIA A100 so the number might slightly change
    )

    input_text = "Hello my name is"
    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello! I'm ChatGLM, an AI assistant. Feel free to tell me your name and share any questions or topics you'd like to discuss.")
    EXPECTED_OUTPUTS.add("Hello! I'm ChatGLM, an AI assistant. I'm here to help you with any questions or topics you'd like to discuss. Please feel free to share more about yourself or ask for assistance on a specific topic.")
    EXPECTED_OUTPUTS.add("\nHello! I'm ChatGLM, an")
    MAX_NEW_TOKENS = 10

    def setUp(self):
        # Models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        quantization_config = TGIQuantizationConfig(load_in_tgi_4bit=True, group_size=64, has_zeros=True)
        self.model_4bit = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, 
                                                               quantization_config=quantization_config, 
                                                               trust_remote_code=True)


class TGI4BitTest(Base4bitTest):
    def setUp(self):
        super().setUp()

        # Models and tokenizer
        self.model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, trust_remote_code=True
        )

    def tearDown(self):
        r"""
        TearDown function needs to be called at the end of each test to free the GPU memory and cache, also to
        avoid unexpected behaviors. Please see: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
        """
        del self.model_fp16
        del self.model_4bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_quantization_config_json_serialization(self):
        r"""
        A simple test to check if the quantization config is correctly serialized and deserialized
        """
        config = self.model_4bit.config

        self.assertTrue(hasattr(config, "quantization_config"))

        _ = config.to_dict()
        _ = config.to_diff_dict()

        _ = config.to_json_string()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        # from bitsandbytes.nn import Params4bit
        from transformers.integrations.tgi import GLMQuantize
        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()
        print(f"mem_fp16: {mem_fp16} / mem_4bit: {mem_4bit} = {mem_fp16 / mem_4bit}")
        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        linear = get_some_linear_layer(self.model_4bit)
        self.assertTrue(isinstance(linear, GLMQuantize))

    def test_original_dtype(self):
        r"""
        A simple test to check if the model succesfully stores the original dtype
        """
        self.assertTrue(hasattr(self.model_4bit.config, "_pre_quantization_dtype"))
        self.assertFalse(hasattr(self.model_fp16.config, "_pre_quantization_dtype"))
        self.assertTrue(self.model_4bit.config._pre_quantization_dtype == torch.float16)

    def test_linear_are_4bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from transformers.integrations import get_keys_to_not_convert
        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()
        modules_to_not_convert = get_keys_to_not_convert(self.model_4bit)
        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in modules_to_not_convert:
                    self.assertTrue(module.weight.dtype == torch.uint8)

    def test_glm_4bit(self):
        r"""
        A simple test to check if 4-bit RWKV inference works as expected.
        """
        text = "Hello my name is"
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        _ = self.model_4bit.generate(input_ids, max_new_tokens=30)


    def test_generate_quality_config(self):
        r"""
        Test that loading the model with the config is equivalent
        """
        query = "Hello my name is"
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                                add_generation_prompt=True,
                                                tokenize=True,
                                                return_tensors="pt",
                                                return_dict=True
                                            )

        inputs = inputs.to("cuda")
        gen_kwargs = {"max_length": 2500, "do_sample": False, "top_k": 1}
        with torch.no_grad():
            outputs = self.model_4bit.generate(**inputs, **gen_kwargs, max_new_tokens=self.MAX_NEW_TOKENS)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            self.assertIn(self.tokenizer.decode(outputs[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after converting it in 8-bit will throw an error.
        Checks also if other models are casted correctly.
        """
        # import pdb
        # pdb.set_trace()
        with self.assertRaises(ValueError):
            # Tries with `str`
            self.model_4bit.to("cpu")

        with self.assertRaises(ValueError):
            # Tries with a `dtype``
            self.model_4bit.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_4bit.to(torch.device("cuda:0"))

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_4bit.float()

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_4bit.half()

        # Test if we did not break anything
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        self.model_fp16 = self.model_fp16.to(torch.float32)
        _ = self.model_fp16.generate(input_ids=encoded_input["input_ids"], max_new_tokens=10)

        # Check this does not throw an error
        _ = self.model_fp16.to("cpu")

        # Check this does not throw an error
        _ = self.model_fp16.half()

        # Check this does not throw an error
        _ = self.model_fp16.float()

    def test_fp32_4bit_conversion(self):
        r"""
        Test whether it is possible to mix both `4bit` and `fp32` weights when using `keep_in_fp32_modules` correctly.
        """
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", load_in_tgi_4bit=True)
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)

    def test_backward(self):
        import torch
        from torch import nn
        import torch.optim as optim
        from transformers.integrations import GLMQuantize, glm_quantize

        x = torch.randn(3, 128, device="cuda",dtype=torch.float16)
        ori_weight = torch.randn(64, 128, device="cuda",dtype=torch.float16)

        linear = nn.Linear(128, 64) # (input_features, output_features)
        linear.weight = nn.Parameter(ori_weight, requires_grad=True)

        linear_glm = GLMQuantize(linear.weight, linear.bias, weight_bit_width=4)
        weight, weight_scale, weight_zero = glm_quantize(linear.weight, weight_bit_width=4)
        linear_glm.weight = weight
        linear_glm.weight_scale = weight_scale
        linear_glm.weight_zero = weight_zero

        linear_glm = linear_glm.to(0)
        x.requires_grad = True
        output = linear_glm(x)
        loss = output.sum()
        output.retain_grad()

        loss.backward()
        normal_output = torch.matmul(output.grad, ori_weight)
        self.assertTrue(x.grad.shape == normal_output.shape)

        print(f"ori_weight: {ori_weight}, \n x.grad: {x.grad}, \n normal_output: {normal_output}")

    def test_dequantize(self):
        def test_dequantize(in_features, out_features, group_size=None, has_zeros=False):
            import torch
            from torch import nn
            import torch.optim as optim
            from transformers.integrations import GLMQuantize, glm_quantize, GLMMatMulBit

            weight = torch.randn(out_features, in_features, device="cuda",dtype=torch.float16)
            weight_bit_width = 4

            linear = nn.Linear(in_features, out_features) # (input_features, output_features)
            linear.weight = nn.Parameter(weight, requires_grad=True)

            weight, weight_scale, weight_zero, ori_weight_zero = glm_quantize(linear.weight, weight_bit_width=weight_bit_width,
                                                            group_size=group_size, has_zeros=has_zeros)
            weight = GLMMatMulBit.dequantize(weight, weight_scale, ori_weight_zero,
                                            weight_bit_width, ori_shape=in_features, group_size=group_size)
            self.assertTrue(weight.shape == linear.weight.shape, f"dequantize shape {weight.shape} not equal ori weight shape {linear.weight.shape}")
            print(f"ori_weight: {linear.weight},\n deq_weight: {weight}")
            print("=======================================================")
        test_dequantize(in_features=128, out_features=68)
        test_dequantize(in_features=129, out_features=68)
        test_dequantize(in_features=128, out_features=68, group_size=64, has_zeros=True)
        test_dequantize(in_features=129, out_features=68, group_size=64, has_zeros=True)
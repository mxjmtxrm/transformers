#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import importlib.util
import json
import math
from dataclasses import dataclass
from math import sqrt
from typing import Dict

from huggingface_hub import hf_hub_download, list_spaces

from ..utils import is_offline_mode
from .python_interpreter import evaluate_python_code
from .tools import TASK_MAPPING, TOOL_CONFIG_FILE, Tool


def custom_print(*args):
    return " ".join(map(str, args))


BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "round": round,
    "sqrt": sqrt,
}


@dataclass
class PreTool:
    name: str
    inputs: Dict[str, type]
    output_type: type
    task: str
    description: str
    repo_id: str


HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB = [
    "image-transformation",
    "text-to-image",
]


def get_remote_tools(logger, organization="huggingface-tools"):
    if is_offline_mode():
        logger.info("You are in offline mode, so remote tools are not available.")
        return {}

    spaces = list_spaces(author=organization)
    tools = {}
    for space_info in spaces:
        repo_id = space_info.id
        resolved_config_file = hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space")
        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)
        task = repo_id.split("/")[-1]
        tools[config["name"]] = PreTool(
            task=task,
            description=config["description"],
            repo_id=repo_id,
            name=task,
            inputs=config["inputs"],
            output_type=config["output_type"],
        )

    return tools


def setup_default_tools(logger):
    default_tools = {}
    main_module = importlib.import_module("transformers")
    tools_module = main_module.tools

    remote_tools = get_remote_tools(logger)
    for task_name, tool_class_name in TASK_MAPPING.items():
        tool_class = getattr(tools_module, tool_class_name)
        default_tools[tool_class.name] = PreTool(
            name=tool_class.name,
            inputs=tool_class.inputs,
            output_type=tool_class.output_type,
            task=task_name,
            description=tool_class.description,
            repo_id=None,
        )

    if not is_offline_mode():
        for task_name in HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB:
            found = False
            for tool_name, tool in remote_tools.items():
                if tool.task == task_name:
                    default_tools[tool_name] = tool
                    found = True
                    break

            if not found:
                raise ValueError(f"{task_name} is not implemented on the Hub.")

    return default_tools


class CalculatorTool(Tool):
    name = "calculator"
    description = "This is a tool that calculates. It can be used to perform simple arithmetic operations."

    inputs = {
        "expression": {
            "type": str,
            "description": "The expression to be evaluated.The variables used CANNOT be placeholders like 'x' or 'mike's age', they must be numbers",
        }
    }
    output_type = str

    def __init__(self):
        import numexpr

        self.numexpr = numexpr

    def __call__(self, expression):
        if isinstance(expression, Dict):
            expression = expression["expression"]
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            self.numexpr.evaluate(
                expression.strip().replace("^", "**"),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return output


class PythonEvaluatorTool(Tool):
    name = "python_evaluator"
    description = "This is a tool that evaluates python code. It can be used to perform calculations. It does not have access to any imports or function definitions."

    inputs = {
        "code": {
            "type": str,
            "description": "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, else you will get an error.",
        }
    }
    output_type = str
    available_tools = BASE_PYTHON_TOOLS.copy()

    def __call__(self, code):
        output = str(evaluate_python_code(code, tools=self.available_tools))
        return output


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem"
    inputs = {"answer": {"type": str, "description": "The final answer to the problem"}}
    output_type = str

    def __call__(self, args):
        return args

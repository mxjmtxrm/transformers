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
import re

from ..utils import cached_file


# docstyle-ignore
CHAT_MESSAGE_PROMPT = """
Human: <<task>>

Assistant: """


DEFAULT_PROMPTS_REPO = "huggingface-tools/default-prompts"
PROMPT_FILES = {"chat": "chat_prompt_template.txt", "run": "run_prompt_template.txt"}


def download_prompt(prompt_or_repo_id, agent_name, mode="run"):
    """
    Downloads and caches the prompt from a repo and returns it contents (if necessary).
    """
    if prompt_or_repo_id is None:
        prompt_or_repo_id = DEFAULT_PROMPTS_REPO

    # prompt is considered a repo ID when it does not contain any kind of space
    if re.search("\\s", prompt_or_repo_id) is not None:
        return prompt_or_repo_id

    prompt_file = cached_file(
        prompt_or_repo_id, PROMPT_FILES[mode], repo_type="dataset", user_agent={"agent": agent_name}
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


DEFAULT_CODE_SYSTEM_PROMPT = """I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignment. You can print intermediate results if it makes sense to do so.
Be sure to provide an 'Code:' token, else the system will be stuck in a loop.

Tools:
<<tool_descriptions>>

---
Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
print(f"The answer is {answer}")
```<end_code>

---
Task: "Identify the oldest person in the `document` and create an image showcasing the result."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator(answer)
```<end_code>

---
Task: "Generate an image using the text given in the variable `caption`."

I will use the following tool: `image_generator` to generate an image.
Code:
```py
image = image_generator(prompt=caption)
```<end_code>

---
Task: "Summarize the text given in the variable `text` and read it out loud."

I will use the following tools: `summarizer` to create a summary of the input text, then `text_reader` to read it out loud.
Code:
```py
summarized_text = summarizer(text)
print(f"Summary: {summarized_text}")
audio_summary = text_reader(summarized_text)
```<end_code>

---
Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

I will use the following tools: `text_qa` to create the answer, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = text_qa(text=text, question=question)
print(f"The answer is {answer}.")
image = image_generator(answer)
```<end_code>

---
Task: "Caption the following `image`."

I will use the following tool: `image_captioner` to generate a caption for the image.
Code:
```py
caption = image_captioner(image)
```<end_code>

---
Above example were using tools that might not exist for you. You only have acces to those Tools:
<<tool_names>>

Now Begin!
Remember to make sure that variables you use are all defined. <<additional_args>>
Be sure to provide an 'Code:' sequence before the code, else you will get an, error.

"""


DEFAULT_REACT_SYSTEM_PROMPT = """Solve the following task as best you can. You have access to the following tools:

<<tool_descriptions>>

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (name of the tool to use) and a `action_input` key (input to the tool).

The value in the "action" field should belong to this list: <<tool_names>>.

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:
Action:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}

Make sure to have the $INPUT as a dictionnary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You will be given:

Task: the task you are given.

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

ALWAYS provide a 'Thought:' and an 'Action:' sequence. You MUST provide at least the 'Action:' sequence to move forward.
You can use the result of the previous action as input for the next action.
The observation will always be a string: it can represent a file, like "image_1.jpg".
Then you can use it as input for the next action. You can do it for instance as follows:

Observation: "image_1.jpg"
Thought: I need to transform the image that I received in the previous observation to make it green.
Action:
{
  "action": "image_transformer",
  "action_input": {"image": "image_1.jpg"}
}

To provide the final answer to the task, use an action blob with "action": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
Action:
{
  "action": "final_answer",
  "action_input": {"answer": "insert your final answer here"}
}

Now begin! <<additional_args>>

"""


DEFAULT_REACT_CODE_SYSTEM_PROMPT = """Solve the following task as best you can. You have access to the following tools:

To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain which tool you will use and for what reason, then in the 'Code:' sequence, you shold write the code in simple Python. The code sequence must end with '/End code' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
It will then be available in the 'Observation:' field, for using this information as input for the next step.

In the end you have to return a final answer using the `final_answer` tool.
Only when you use function final_answer() will your final answer be returned.

You have access to the following tools:
<<tool_descriptions>>

You will be provided a task:
Task: the task you are given.

Then you should answer as follows:

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.
Code:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
answer = image_qa(image=image, question=translated_question)
final_answer(f"The answer is {answer}")
```<end_code>

The 'Action input:' should be the input for the tool you are using.
Make sure to have the $ACTION_INPUT as a dictionnary, with the right keys for the tool you are using, and do not put variable names as input if you know the right values.

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Code:
```py
answer = document_qa(document=document, question="What is the oldest person?")
print(answer)
```<end_code>
Observation: "The oldest person in the document is John Doe."

Thought: I will now generate an image showcasing the oldest person.

Code:
```py
image = image_generator("An image of John Doe, an old man.")
final_answer(image)
```<end_code>

Example:::
Task: "Generate an image using the text given in the variable `caption`."

Thought: I will use the following tool: `image_generator` to generate an image.
Code:
```py
image = image_generator(prompt=caption)
final_answer(image)
```<end_code>

Example:::
Task: "Summarize the text given in the variable `text` and read it out loud."

Thought: I will use the following tools: `summarizer` to create a summary of the input text, then `text_reader` to read it out loud.

Code:
```py
summarized_text = summarizer(text)
print(f"Summary: {summarized_text}")
```<end_code>

Observation: "Summary: 'The text is about the history of the Roman Empire.'"

Thought: I will now read the summarized text out loud.

Code:
```py
audio_summary = text_reader(summarized_text)
final_answer(audio_summary)
```<end_code>

Example:::
Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

Thought: I will use the following tools: `text_qa` to create the answer, then `image_generator` to generate an image according to the answer.

Be sure to provide an 'Code:' token, else the system will be stuck in a loop.

Code:
```py
answer = text_qa(text=text, question=question)
image = image_generator(answer)
final_answer(image)
```<end_code>

Example:::
Task: "What is the result of the following operation: 5 + 3 + 1298987654.6789098765?"

Thought: I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool

Code:
```py
result = 5 + 3 + 1298987654.6789098765
final_answer(result)
```<end_code>

Example:::
Task: "Which city has the highest population , Guangzhou or Shanghai?"

Thought: I will use the tool `search` to get the population of both cities.
Code:
```py
population_guangzhou = search("Guangzhou population")
print("Population Guangzhou:", population_guangzhou)
population_shanghai = search("Shanghai population")
print("Population Shanghai:", population_shanghai)
```<end_code>
Observation:
Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '24 million'

Thought: I know that Shanghai has the highest population.
Code:
```py
final_answer("Shanghai")
```<end_code>

Example:::
Task: "What is the current age of the pope, raised to the power 0.36?"

Thought: I will use the tool `search` to get the age of the pope, then raise it to the power 0.36.
Code:
```py
pope_age = search(query="current pope age")
print("Pope age:", pope_age)
```<end_code>

Observation:
Pope age: "The pope Francis is currently 85 years old."

Thought: I know that the pope is 85 years old. Let's compute the result using python code.
Code:
```py
pope_current_age = 85 ** 0.36
final_answer(pope_current_age)
```<end_code>


Above example were using tools that might not exist for you. You only have acces to those tools:
<<tool_names>>
You also can perform computations in the python code you generate.

Always provide a 'Thought:' and an 'Code:\n```py' sequence ending with '```<end_code>' sequence. You MUST provide at least the 'Code:' sequence to move forward.

Remember to not perform too many operations in a single code block! You should split the task into intermediate code blocks. Then use one single print() at the end of each step to save the intermediate result. Then use final_answer() to return the final result.

DO NOT pass the arguments as a dict as in 'answer = ask_search_agent({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = ask_search_agent(query="What is the place where James Bond lives?")'.

Now Begin! <<additional_args>>
"""

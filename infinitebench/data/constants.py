# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


TASKS = {
    "passkey": {
        "tokens_to_generate": 6,
        "context_template": """There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n""",
        "query_template": """{input}\n\nThe pass key is""",
        "answer_prefix": """""",
    },
    "number_string": {
        "tokens_to_generate": 12,
        "context_template": """There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n""",
        "query_template": """{input}\n\nThe sequence of digits is""",
        "answer_prefix": """""",
    },
    "kv_retrieval": {
        "tokens_to_generate": 50,
        "context_template": """Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n""",
        "query_template": """{input}""",
        "answer_prefix": """""",
    },
    "longbook_sum_eng": {
        "tokens_to_generate": 1200,
        "context_template": """Summarize the book below.\n\n{context}\n\n""",
        "query_template": """Summary:""",
        "answer_prefix": """""",
    },
    "longbook_choice_eng": {
        "tokens_to_generate": 40,
        "context_template": """Read the book and answer the question.\n\n{context}\n\n""",
        "query_template": """Question: {input}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is""",
        "answer_prefix": """""",
    },
    "longbook_qa_eng": {
        "tokens_to_generate": 40,
        "context_template": """Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\n""",
        "query_template": """Question: {input}\nAnswer:""",
        "answer_prefix": """""",
    },
    "longbook_qa_chn": {
        "tokens_to_generate": 40,
        "context_template": """阅读以下书籍然后回答问题。\n\n{context}\n\n""",
        "query_template": """问题: {input}\n答案: """,
        "answer_prefix": """""",
    },
    "math_find": {
        "tokens_to_generate": 3,
        "context_template": """{prefix}\n\n{context}\n\n""",
        "query_template": """{input}""",
        "answer_prefix": """""",
    },
    "code_debug": {
        "tokens_to_generate": 5,
        "context_template": """Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\n""",
        "query_template": """Options:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:""",
        "answer_prefix": """""",
    },
    "longdialogue_qa_eng": {
        "tokens_to_generate": 40,
        "context_template": """Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", and you should try to guess who that character is.\n\n{context}\n\n""",
        "query_template": """The name that has been replaced with $$MASK$$ is likely""",
        "answer_prefix": """""",
    },
}

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

"""
Prepare jsonl with field `input_context`, `input_query` and `outputs`.
{
    "index" int,
    "input_context": str,
    "input_query": str,
    "outputs": [str],
}

Usage:
python infinitebench/data/prepare.py \
    --tokenizer_path <path_to_hf_tokenizer> \
    --max_seq_length <dataset_sequence_length> \
    --model_template_type <prompt_template>
"""

import argparse
import os
import json
import time
import re

from tqdm import tqdm

from constants import TASKS
from tokenizer import select_tokenizer
from template import PROMPT_TEMPLATES


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))


def download_dataset(task_name, output_file):
    if not os.path.exists(output_file):
        print(f'Downloading {task_name} dataset...')
        url = f'https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/{task_name}.jsonl'
        os.system(f'wget -P {os.path.dirname(output_file)} {url}')


def process_math_find(sample, context_template, query_template):
    query, context = sample['input'], sample['context']
    # Find "the * number" from the query
    find_result = re.findall(r'The .+ of', query)
    assert find_result, f'Cannot find the target number in {query}'
    target_number = find_result[0].lower()[:-3]
    # Replace the number with the answer
    prefix = f'What is {target_number} in the following list?'
    processed_context = context_template.format(prefix=prefix, context=context)
    processed_query = query_template.format(input=query)
    return processed_context, processed_query


def prepare_task_prompt(sample, task_name, task_config):
    context_template = task_config['context_template']
    query_template = task_config['query_template']
    answer_prefix_template = task_config['answer_prefix']

    option_a, option_b, option_c, option_d = '', '', '', ''
    if len(sample['options']) > 0:
        if len(sample['options']) != 4:
            raise ValueError(f'options should be 4, but got {len(sample["options"])}')
        option_a, option_b, option_c, option_d = sample['options']

    if task_name == 'math_find':
        context, query = process_math_find(sample, context_template, query_template)
        answer_prefix = answer_prefix_template
    else:
        context = context_template.format(context=sample['context'])
        query = query_template.format(
            input=sample['input'],
            OPTION_A=option_a,
            OPTION_B=option_b,
            OPTION_C=option_c,
            OPTION_D=option_d,
        )
        answer_prefix = answer_prefix_template

    if answer_prefix and query.endswith(answer_prefix):
        query = query[: -len(answer_prefix)]

    return context, query, answer_prefix


def convert_to_prompt_template(context, query, answer_prefix, prompt_template):
    model_template_context, model_template_query = prompt_template.split('{task_template}')
    context_template = model_template_context + context
    query_template = query + model_template_query + answer_prefix
    return context_template, query_template


def truncate_sample_text(
    context, query, answer_prefix, prompt_template, tokens_to_generate, max_seq_length, tokenizer
):
    input_context, input_query = convert_to_prompt_template(context, query, answer_prefix, prompt_template)
    num_tokens = len(tokenizer.text_to_tokens(input_context + input_query)) + tokens_to_generate

    # If prompt is long, remove from middle of context
    if num_tokens > max_seq_length:
        tokens_to_remove = num_tokens - max_seq_length + 1

        context_tokens = tokenizer.text_to_tokens(context)
        split_idx = len(context_tokens) // 2

        # Handle odd number of tokens to remove by calculating left and right amounts
        left_remove = tokens_to_remove // 2
        right_remove = tokens_to_remove - left_remove

        context_tokens = context_tokens[0 : split_idx - left_remove] + context_tokens[split_idx + right_remove :]
        context = tokenizer.tokens_to_text(context_tokens)

        input_context, input_query = convert_to_prompt_template(context, query, answer_prefix, prompt_template)
        num_tokens = len(tokenizer.text_to_tokens(input_context + input_query)) + tokens_to_generate

    assert num_tokens <= max_seq_length, f'num_tokens ({num_tokens}) > max_seq_length ({max_seq_length})'
    return input_context, input_query, num_tokens


def get_answer(sample, task_name):
    if task_name in ['code_debug', 'longbook_choice_eng']:
        options = 'ABCD'
        if isinstance(sample['answer'], str):
            ret = [sample['answer'], options[sample['options'].index(sample['answer'])]]
        elif isinstance(sample['answer'], list):
            if len(sample['answer']) == 1:
                ret = [sample['answer'][0], options[sample['options'].index(sample['answer'][0])]]
            elif len(sample['answer']) == 2 and sample['answer'][1] in ['A', 'B', 'C', 'D']:
                ret = sample['answer']
            else:
                raise ValueError
        else:
            raise ValueError
        return ret

    return sample['answer']


def process_task_file(task_name, task_config, prompt_template, tokenizer, max_seq_length, output_dir):
    source_file = os.path.join(CURRENT_DIR, 'task_files', f'{task_name}.jsonl')
    download_dataset(task_name, source_file)
    output_file = os.path.join(output_dir, f'{task_name}.jsonl')

    num_output_samples = 0
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file) as f:
            num_output_samples = sum(1 for _ in f)

    with open(source_file) as fread:
        with open(output_file, 'a' if num_output_samples > 0 else 'w') as fout:
            print(f'\rSamples Processed: {num_output_samples}', end='\r')
            for idx, line in enumerate(fread):
                if idx < num_output_samples:
                    continue

                sample = json.loads(line)

                context, query, answer_prefix = prepare_task_prompt(sample, task_name, task_config)
                input_context, input_query, num_tokens = truncate_sample_text(
                    context,
                    query,
                    answer_prefix,
                    prompt_template,
                    task_config['tokens_to_generate'],
                    max_seq_length,
                    tokenizer,
                )

                fout.write(
                    json.dumps(
                        {
                            'index': idx,
                            'input_context': input_context,
                            'input_query': input_query,
                            'outputs': get_answer(sample, task_name),
                            'length': num_tokens,
                        }
                    )
                    + '\n'
                )
                print(f'\rSamples Processed: {idx + 1}', end='\r')
            print('\n')


def prepare_dataset(output_dir, tokenizer_path, tokenizer_type, max_seq_length, model_template_type):
    start_time = time.time()

    prompt_template = PROMPT_TEMPLATES[model_template_type]['template']
    tokenizer = select_tokenizer(tokenizer_type, tokenizer_path)

    for task, task_config in TASKS.items():
        print(f'Processing {task}...')
        process_task_file(task, task_config, prompt_template, tokenizer, max_seq_length, output_dir)

    print(f'Used time: {round((time.time() - start_time) / 60, 1)} minutes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=os.path.join(ROOT_DIR, 'dataset'), help='dataset folder to save dataset')
    parser.add_argument('--tokenizer_path', required=True, help='path to the tokenizer model')
    parser.add_argument('--tokenizer_type', default='hf', help='[Options] hf')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        required=True,
        help='max sequence length including all input tokens and generated tokens.',
    )
    parser.add_argument('--model_template_type', required=True, help='Options in `template.py`')
    args = parser.parse_args()

    assert (
        args.model_template_type in PROMPT_TEMPLATES
    ), f'{args.model_template_type} is not found in {PROMPT_TEMPLATES.keys()}'

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f'Tokenizer path {args.tokenizer_path} does not exist.')

    args.save_dir = os.path.join(args.save_dir, f'infinitebench_{args.model_template_type}_{args.max_seq_length}')
    os.makedirs(args.save_dir, exist_ok=True)

    prepare_dataset(
        args.save_dir, args.tokenizer_path, args.tokenizer_type, args.max_seq_length, args.model_template_type
    )

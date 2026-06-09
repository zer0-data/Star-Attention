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

import argparse
import json
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_LIST = [
    'longbook_sum_eng',
    'longbook_qa_eng',
    'longbook_choice_eng',
    'longdialogue_qa_eng',
    'longbook_qa_chn',
    'code_debug',
    'math_find',
    'passkey',
    'number_string',
    'kv_retrieval',
]


def parse_scores(scores_path, task_num_samples):
    with open(scores_path) as f:
        scores = json.load(f)

    accuracies, unfinished_tasks = [], []
    for task in TASK_LIST:
        if task not in scores:
            accuracies.append(-1)
            unfinished_tasks.append(task)
        else:
            accuracies.append(scores[task]['acc'])
            if scores[task]['num_samples'] < task_num_samples[task]:
                unfinished_tasks.append(task)

    return accuracies, unfinished_tasks


def resubmit_experiments(seq_result_dir, tasks):
    submit_error_tasks = []
    for i in os.listdir(os.path.join(seq_result_dir, 'logs')):
        if i in tasks:
            task_submission_script = os.path.join(seq_result_dir, 'logs', i, 'generate_predictions.sh')
            if os.path.exists(task_submission_script):
                print('Resubmitting:', f'{os.path.basename(seq_result_dir)} - {i}')
                os.system(f'bash {task_submission_script}')
            else:
                submit_error_tasks.append(i)

    os.system(f'bash {os.path.join(seq_result_dir, "logs", "evaluate_responses.sh")}')

    if submit_error_tasks:
        print(f'++ Submission errors: {", ".join(submit_error_tasks)} ++')


def gather_experiment_results(exp_dir, submit=False):
    with open(os.path.join(BASE_DIR, 'infinitebench', 'task_config.yaml')) as f:
        task_config = yaml.safe_load(f)
    task_num_samples = {k: v['num_samples'] for k, v in task_config.items()}

    missing_seq_len_exps = []
    seq_len_results = {}
    for seq_len in sorted([x for x in os.listdir(exp_dir) if x.isdigit()], key=int):
        seq_len_dir = os.path.join(exp_dir, seq_len)
        if not os.path.exists(os.path.join(seq_len_dir, 'scores.json')):
            missing_seq_len_exps.append(seq_len)
            continue

        accuracies, unfinished_tasks = parse_scores(os.path.join(seq_len_dir, 'scores.json'), task_num_samples)
        seq_len_results[seq_len] = {
            'accuracies': accuracies,
            'unfinished_tasks': unfinished_tasks,
        }

        if submit and unfinished_tasks:
            resubmit_experiments(seq_len_dir, unfinished_tasks)

    if missing_seq_len_exps:
        print(f'\n++ Missing sequence length experiments: {", ".join(missing_seq_len_exps)} ++')

    print('\nFull CSV:')
    print('seq_len,' + ','.join(TASK_LIST))
    for k, v in seq_len_results.items():
        accuracies = [f'{x * 100:.2f}' for x in v['accuracies']]
        print(f'{",".join([k] + accuracies)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--exp',
        required=True,
        help=(
            'experiment name containing the results or path to the results directory. '
            'If experiment name is given then it will be searched in `output_dir`. '
            '`exp` is given more priority than `project`.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join(BASE_DIR, 'results'),
        help='results directory',
    )
    parser.add_argument('-s', '--submit', action='store_true', help='re-submit the missing experiments')
    args = parser.parse_args()

    if '/' not in args.exp:
        args.exp = os.path.join(args.output_dir, args.exp)
    if not os.path.isdir(args.exp):
        raise NotADirectoryError(f'Invalid experiment path: {args.exp}')

    gather_experiment_results(args.exp, submit=args.submit)

import argparse
import os
import json
import re
import string
from collections import Counter

from tqdm import tqdm
import evaluate


ROUGE_SCORER = evaluate.load("rouge")
TASK_LIST = [
    'passkey',
    'number_string',
    'kv_retrieval',
    'longdialogue_qa_eng',
    'longbook_sum_eng',
    'longbook_choice_eng',
    'longbook_qa_eng',
    'longbook_qa_chn',
    'math_find',
    'code_debug',
]


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Chinese version. Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth) -> tuple[float, float, float]:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths) -> float:
    """Computes the F1, recall, and precision."""
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: list[str]) -> float:
    """
    QA F1 score for chinese.
    """
    f1 = 0
    prec = 0
    recall = 0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(ground_truth)

        # One character one token.
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1, this_prec, this_recall = scores
        f1 = max(f1, this_f1)
        prec = max(prec, this_prec)
        recall = max(recall, this_recall)
    return f1


def load_json(fname):
    return json.load(open(fname))


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r", encoding="utf8") as fin:
        for line in fin:
            if line.strip() == "":  # Skip empty lines
                continue
            if i == cnt:
                break
            if line.strip() == "":  # Skip empty lines
                continue
            yield json.loads(line)
            i += 1


def first_int_match(prediction):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def split_retrieval_answer(pred: str):
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return words


def get_score_one_kv_retrieval(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    for c in ['\n', ':', '\"', '\'', '.', ',', '?', '!', '{', '}']:
        pred = pred.replace(c, ' ')
    words = pred.split()
    return label in words


def get_score_one_passkey(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_number_string(pred, label) -> bool:
    if isinstance(label, list):
        label = label[0]
    return label == first_int_match(pred)


def get_score_one_code_debug(pred, label) -> bool:
    """
    Returns the score of one example in Code.Debug.
    """
    pred = pred.strip()
    label_c = label[1]
    fn_name = label[0]
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred == label_c:
            return True
    ans_prefixes = [
        "answer is:",
        # "answer is",
        # "error is",
        "is:",
        "answer:",
        "correct option is:",
    ]
    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    if pred.startswith(label_c) or pred.startswith(fn_name):
        return True
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        pred = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if pred.startswith(s):
                return True
        return False
    return False


def get_score_one_math_find(pred, label) -> bool:
    if isinstance(label, list):
        # In math_find, there is always only one label.
        label = label[0]
    if isinstance(label, int):
        # Find first int or float
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        return int(first_num) == label
    elif isinstance(label, float):
        # Find first float or int
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        return float(first_float) == label
    else:
        raise TypeError(f"Expected int or float, got {type(label)}")


def get_score_one_longdialogue_qa_eng(pred, label) -> bool:
    pred = pred.strip()
    pred = pred.upper()
    for item in label:
        if item.upper() in pred:
            return True
    return False


def get_score_one_longbook_choice_eng(pred, label) -> bool:
    # Just use the first letter as the prediction
    pred = pred.strip()
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"

    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, ' ')
    while '  ' in pred:
        pred = pred.replace('  ', ' ')
    ans_prefixes = [
        'answer is:',
        'answer:',
        'answer is',
        'option is',
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


def get_score_one_longbook_qa_eng(pred, label) -> float:
    return qa_f1_score(pred, label)


def get_score_one_longbook_sum_eng(pred: str, label: str) -> float:
    score = ROUGE_SCORER.compute(predictions=[pred], references=[label], use_aggregator=False)
    return score["rougeLsum"][0]  # type: ignore


def get_score_one_longbook_qa_chn(pred, label) -> float:
    return qa_f1_score_zh(pred, label)


def get_score_one(pred: str, label: str, task_name: str) -> float:
    """
    Computes the score for one prediction.
    Returns one float (zero and one for boolean values).
    """
    NAME_TO_SCORE_GETTER = {
        # Retrieve
        "kv_retrieval": get_score_one_kv_retrieval,
        "kv_retrieval_prefix": get_score_one_kv_retrieval,
        "kv_retrieval_both": get_score_one_kv_retrieval,
        "passkey": get_score_one_passkey,
        "number_string": get_score_one_number_string,
        # Code
        "code_debug": get_score_one_code_debug,
        # Longbook
        "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": get_score_one_longbook_choice_eng,
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        # Math
        "math_find": get_score_one_math_find,
    }
    assert task_name in NAME_TO_SCORE_GETTER, f'Invalid task name: {task_name}'
    score = NAME_TO_SCORE_GETTER[task_name](pred, label)
    return float(score)


def get_labels(preds: list) -> list[str]:
    return [x.get('outputs', "XXXXXXXXXX") for x in preds]


def get_preds(preds: list) -> list[str]:
    return [pred['pred'] for pred in preds]


def get_score(labels: list, preds: list, task_name: str) -> float:
    """Computes the average score for a task"""
    assert len(labels) == len(preds)
    scores = []
    for label, pred in tqdm(zip(labels, preds)):
        score = get_score_one(pred, label, task_name)
        scores.append(score)
    return sum(scores) / len(scores)


def compute_task_scores(preds_path, task_name):
    print('Loading prediction results from', preds_path)
    preds = list(iter_jsonl(preds_path))
    labels = get_labels(preds)
    preds = get_preds(preds)

    acc = get_score(labels, preds, task_name)
    return acc, len(labels)


def compute_scores(preds_path):
    task_scores = {}
    for task in TASK_LIST:
        predictions_path = os.path.join(preds_path, f'{task}.jsonl')
        if not os.path.exists(predictions_path):
            print(f'Predictions for {task} not found. Skipping...')
            continue
        acc, num_samples = compute_task_scores(predictions_path, task)
        task_scores[task] = {
            'acc': acc,
            'num_samples': num_samples,
        }

    output_file = os.path.join(preds_path, 'scores.json')
    with open(output_file, 'w') as f:
        json.dump(task_scores, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True, help='path containing the predictions')
    args = parser.parse_args()

    compute_scores(args.result_dir)

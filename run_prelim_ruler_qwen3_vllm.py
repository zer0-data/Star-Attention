"""Dense full-attention RULER baseline for Qwen3-4B-Instruct-2507 via vLLM.

Companion to ``run_prelim_ruler_qwen3.py`` — same dataset prep, same prompt
template, same RULER scoring — but inference is done with vLLM (flash
attention 2 + PagedAttention + continuous batching) over the *full*
context. No star attention, no blockwise KV-cache, no summaries.

Use this to establish the upper-bound dense accuracy that the star sweep
is being compared against, and to get those numbers fast (single-pass
batched generation instead of the per-sample serial loop in the star
script).

Example
-------
  python run_prelim_ruler_qwen3_vllm.py \\
    --model_path Qwen/Qwen3-4B-Instruct-2507 \\
    --seq_lengths 16384,32768 \\
    --tasks niah_single_1,niah_multikey_1,qa_1 \\
    --num_samples 100 \\
    --max_model_len 65536 \\
    --gpu_memory_utilization 0.9

Dependencies
------------
  pip install vllm>=0.5.4
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import Dict, List

import yaml

from ruler import PROMPT_TEMPLATES
from ruler.eval.synthetic.constants import TASKS as METRIC_TASKS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Helpers (copied from run_prelim_ruler_qwen3.py for self-containment)
# --------------------------------------------------------------------------- #

def _csv(values: str) -> List[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _csv_int(values: str) -> List[int]:
    return [int(v) for v in _csv(values)]


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_task_configs() -> Dict[str, dict]:
    with open(os.path.join(BASE_DIR, "ruler", "synthetic_task_config.yaml")) as f:
        tasks_customized = yaml.safe_load(f)
    return {
        name: {
            "category": cfg["task"],
            "metric_fn": METRIC_TASKS[cfg["task"]]["metric_fn"],
        }
        for name, cfg in tasks_customized.items()
    }


def _load_tokens_to_generate() -> Dict[str, int]:
    with open(os.path.join(BASE_DIR, "ruler", "synthetic_inference_config.yaml")) as f:
        return yaml.safe_load(f)["tokens_to_generate"]


def _ensure_dataset(
    *,
    data_root: str,
    seq_length: int,
    task: str,
    tokenizer_path: str,
    prompt_config: str,
    num_samples: int,
    force_regen: bool,
) -> str:
    seq_dir = os.path.join(data_root, str(seq_length))
    out_file = os.path.join(seq_dir, task, "validation.jsonl")
    if os.path.exists(out_file) and not force_regen:
        return out_file
    os.makedirs(seq_dir, exist_ok=True)
    cmd = (
        f"python ruler/data/prepare.py "
        f"--save_dir {seq_dir} "
        f"--task {task} "
        f"--tokenizer_path {tokenizer_path} "
        f"--tokenizer_type hf "
        f"--max_seq_length {seq_length} "
        f"--model_template_type {prompt_config} "
        f"--num_samples {num_samples}"
    )
    print(f"    [data] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=BASE_DIR, check=False)
    if res.returncode != 0 or not os.path.exists(out_file):
        raise RuntimeError(
            f"Failed to generate dataset for seq_length={seq_length} task={task}; "
            f"expected {out_file}"
        )
    return out_file


# --------------------------------------------------------------------------- #
# One cell — batched vLLM inference + scoring
# --------------------------------------------------------------------------- #

def run_cell(
    llm,
    sampling_params_cls,
    samples: List[dict],
    *,
    task: str,
    metric_fn,
    tokens_to_generate: int,
    stop_words: List[str],
    seq_length: int,
    start_sample_index: int,
    num_samples: int,
    results_file: str,
    predictions_dir: str,
    model_tag: str,
):
    end_idx = min(start_sample_index + num_samples, len(samples))
    cell_samples = samples[start_sample_index:end_idx]
    total = len(cell_samples)
    if total == 0:
        return

    print(
        f"\n>>> seq_length={seq_length} task={task} "
        f"max_new_tokens={tokens_to_generate} -> {total} samples (batched)"
    )

    prompts = [s["input_context"] + s["input_query"] for s in cell_samples]

    params = sampling_params_cls(
        temperature=0.0,                 # greedy → deterministic eval
        top_p=1.0,
        max_tokens=tokens_to_generate,
        stop=stop_words or None,
    )

    # Single batched call — vLLM schedules them concurrently
    vllm_outputs = llm.generate(prompts, params)

    # vLLM may reorder; map by request_id which matches input index
    # (LLM.generate preserves input order in returned list)
    preds: List[str] = [o.outputs[0].text for o in vllm_outputs]
    refs: List[List[str]] = [
        s.get("outputs", [s.get("output", "")]) for s in cell_samples
    ]

    cell_pred_path = os.path.join(
        predictions_dir, f"{seq_length}__{task}__dense_vllm.jsonl"
    )
    os.makedirs(predictions_dir, exist_ok=True)
    with open(cell_pred_path, "w", encoding="utf-8") as fout:
        for sample, pred, ref in zip(cell_samples, preds, refs):
            fout.write(json.dumps({
                "index": sample.get("index", -1),
                "pred": pred,
                "input_context": sample["input_context"],
                "input_query": sample["input_query"],
                "outputs": ref,
                "others": sample.get("others", {}),
                "truncation": sample.get("truncation", -1),
                "length": sample.get("length", -1),
            }) + "\n")

    # Some RULER tasks (e.g. cwe) emit samples with outputs=[]; skip them.
    filtered = [(p, r) for p, r in zip(preds, refs) if r]
    n_skipped = len(preds) - len(filtered)
    if n_skipped:
        print(f"    [warn] skipping {n_skipped} sample(s) with empty refs")
    if not filtered:
        print(f"    [warn] no scorable samples; skipping cell")
        return
    preds_s, refs_s = zip(*filtered)
    score = metric_fn(list(preds_s), list(refs_s))
    n_null = sum(1 for p in preds_s if len(p) == 0)
    print(
        f"    === seq={seq_length}/{task}: "
        f"score={score:.2f} ({n_null} nulls / {len(preds)}) ==="
    )

    results_line = (
        f"{datetime.datetime.utcnow().isoformat()} | "
        f"model={model_tag} | "
        f"attn=dense_vllm_full | "
        f"seq_length={seq_length} | "
        f"task={task} | "
        f"tokens_to_generate={tokens_to_generate} | "
        f"samples=start:{start_sample_index},n:{num_samples} | "
        f"nulls={n_null}/{len(preds)} | "
        f"score={score:.2f}\n"
    )
    try:
        with open(results_file, "a", encoding="utf-8") as rf:
            rf.write(results_line)
    except Exception as e:
        print(f"    Failed to write results: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    tasks = _csv(args.tasks)
    seq_lengths = _csv_int(args.seq_lengths)

    task_configs = _load_task_configs()
    tokens_to_generate_map = _load_tokens_to_generate()
    for t in tasks:
        if t not in task_configs:
            raise ValueError(
                f"Unknown RULER task {t!r}; valid: {list(task_configs)}"
            )
        if t not in tokens_to_generate_map:
            raise ValueError(
                f"No tokens_to_generate for task {t!r}"
            )

    if args.prompt_config not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt_config {args.prompt_config!r}; "
            f"valid: {list(PROMPT_TEMPLATES)}"
        )
    stop_words = PROMPT_TEMPLATES[args.prompt_config]["stop_words"] or []

    # vLLM imports deferred so the script can be argparse-introspected
    # without vllm installed (e.g. when running with --help).
    from vllm import LLM, SamplingParams

    print("=" * 72)
    print("  RULER Dense Full-Attention Baseline — vLLM (Qwen3-4B-Instruct-2507)")
    print("=" * 72)
    print(f"  Model            : {args.model_path}")
    print(f"  Prompt config    : {args.prompt_config}")
    print(f"  Stop words       : {stop_words}")
    print(f"  Seq lengths      : {seq_lengths}")
    print(f"  Tasks            : {tasks}")
    print(f"  samples/cell     : {args.num_samples} "
          f"(start_index={args.start_sample_index})")
    print(f"  max_model_len    : {args.max_model_len}")
    print(f"  gpu_mem_util     : {args.gpu_memory_utilization}")
    print(f"  tensor_parallel  : {args.tensor_parallel_size}")
    print(f"  data_dir         : {args.data_dir}")
    print(f"  predictions_dir  : {args.predictions_dir}")
    print(f"  results_file     : {args.results_file}")
    print(f"  total cells      : {len(seq_lengths) * len(tasks)}")
    print("=" * 72)

    # --- 1. Prepare datasets ---
    print("\n--- 1. Preparing datasets ---")
    samples_by_key: Dict[tuple, List[dict]] = {}
    for seq_length in seq_lengths:
        for task in tasks:
            data_path = _ensure_dataset(
                data_root=args.data_dir,
                seq_length=seq_length,
                task=task,
                tokenizer_path=args.model_path,
                prompt_config=args.prompt_config,
                num_samples=args.num_samples + args.start_sample_index,
                force_regen=args.force_regen,
            )
            samples = _read_jsonl(data_path)
            print(f"    {seq_length}/{task}: {len(samples)} samples ({data_path})")
            samples_by_key[(seq_length, task)] = samples

    # --- 2. Load vLLM engine ONCE ---
    # vLLM uses flash attention 2 + PagedAttention by default on Ampere+.
    # bfloat16 matches the model's native dtype (torch_dtype=bfloat16 in config).
    print("\n--- 2. Loading vLLM engine ---")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
        enforce_eager=args.enforce_eager,
        # vLLM default attention backend is FlashAttention on supported GPUs.
        # Override via VLLM_ATTENTION_BACKEND env var if needed.
    )

    model_tag = os.path.basename(args.model_path.rstrip("/"))

    # --- 3. Run cells ---
    print("\n--- 3. Evaluating ---")
    total_cells = len(seq_lengths) * len(tasks)
    cell_idx = 0
    for seq_length in seq_lengths:
        for task in tasks:
            cell_idx += 1
            print(f"\n--- [{cell_idx}/{total_cells}] ---")
            run_cell(
                llm=llm,
                sampling_params_cls=SamplingParams,
                samples=samples_by_key[(seq_length, task)],
                task=task,
                metric_fn=task_configs[task]["metric_fn"],
                tokens_to_generate=tokens_to_generate_map[task],
                stop_words=stop_words,
                seq_length=seq_length,
                start_sample_index=args.start_sample_index,
                num_samples=args.num_samples,
                results_file=args.results_file,
                predictions_dir=args.predictions_dir,
                model_tag=model_tag,
            )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model ID or local path to a Qwen3 checkpoint.",
    )
    parser.add_argument(
        "--prompt_config",
        default="qwen3",
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Prompt template key (sets ChatML wrapping + stop words).",
    )
    parser.add_argument(
        "--seq_lengths",
        default="16384,32768",
        help=(
            "CSV list of context sequence lengths. Qwen3-4B-Instruct-2507 "
            "supports up to 262,144 tokens natively."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="niah_single_1,niah_multikey_1,qa_1",
        help="CSV list of RULER task names.",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_sample_index", type=int, default=0)
    parser.add_argument(
        "--max_model_len", type=int, default=65536,
        help=(
            "Max sequence length vLLM allocates KV cache for. Must be >= "
            "the largest seq_length you run + tokens_to_generate. Bump up "
            "to 262144 if you need the full context window."
        ),
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="Fraction of GPU memory vLLM may use for weights + KV cache.",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--enforce_eager", action="store_true",
        help="Disable CUDA graphs (slower, but works on more setups).",
    )
    parser.add_argument(
        "--force_regen", action="store_true",
        help="Regenerate RULER datasets even if cached.",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(BASE_DIR, "dataset", "prelim_ruler_qwen3_4b_instruct_2507"),
        help="Root for cached RULER data (shared with star script).",
    )
    parser.add_argument(
        "--predictions_dir",
        default=os.path.join(
            BASE_DIR, "results", "prelim_ruler_qwen3_4b_instruct_2507_vllm", "predictions"
        ),
    )
    parser.add_argument(
        "--results_file",
        default=os.path.join(
            BASE_DIR, "prelim_ruler_qwen3_4b_instruct_2507_vllm_accuracies.txt"
        ),
    )
    args = parser.parse_args()
    main(args)

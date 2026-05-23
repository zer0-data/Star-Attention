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

"""Timing script for Star Attention vs Dense vs Statistical Cross-Attention.

Measures wall-clock inference time across three modes:
  dense   -- full attention, single process (torchrun --nproc_per_node=1)
  star    -- anchor mode (--anchor_block_size > 0, --summary_chunks 0)
  star    -- statistical mode (--anchor_block_size -1, --summary_method max_idf etc.)

Warmup runs sample[0] --warmup_rounds times before the timer starts.
Per-sample times are printed + a summary table at the end.

Example (dense, 1 GPU):
  torchrun --nproc_per_node=1 scripts/time_inference.py \\
      --model_path /model --attn_type dense \\
      --prompt_config llama3 \\
      --input_path /data/timing_64k.jsonl

Example (star anchor, 4 GPUs):
  torchrun --nproc_per_node=4 scripts/time_inference.py \\
      --model_path /model --attn_type star \\
      --prompt_config llama3 \\
      --block_size 16384 --anchor_block_size 16384 --summary_chunks 0 \\
      --input_path /data/timing_64k.jsonl

Example (statistical max_idf, 4 GPUs):
  torchrun --nproc_per_node=4 scripts/time_inference.py \\
      --model_path /model --attn_type star \\
      --prompt_config llama3 \\
      --block_size 16384 --anchor_block_size -1 \\
      --summary_method max_idf --summary_chunks 4 --chunk_size 32 --sink_size 64 \\
      --input_path /data/timing_64k.jsonl
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import torch.distributed as dist

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ruler import PROMPT_TEMPLATES


def read_jsonl(filename: str, num_lines: int = -1) -> List[dict]:
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if num_lines > 0 and i + 1 >= num_lines:
                break
    return lines


def init_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_model(
    model_path: str,
    attn_type: str,
    tokens_to_generate: int,
    stop_words: Optional[List[str]],
    block_size: int,
    anchor_block_size: int,
    summary_chunks: int,
    chunk_size: int,
    summary_method: str,
    sink_size: int,
    discard_summary_kv: bool,
):
    if attn_type == "dense":
        from model import DenseAttentionModel
        return DenseAttentionModel(
            path=model_path,
            max_new_tokens=tokens_to_generate,
            stop_words=stop_words,
        )

    if attn_type == "ring":
        from model import RingAttentionModel
        return RingAttentionModel(
            path=model_path,
            max_new_tokens=tokens_to_generate,
            stop_words=stop_words,
        )

    if attn_type == "star":
        from model import StarAttentionModel
        assert block_size > 0, "--block_size required for star attention"
        return StarAttentionModel(
            path=model_path,
            block_size=block_size,
            max_new_tokens=tokens_to_generate,
            stop_words=stop_words,
            anchor_block_size=anchor_block_size,
            summary_chunks=summary_chunks,
            chunk_size=chunk_size,
            summary_method=summary_method,
            sink_size=sink_size,
            discard_summary_kv=discard_summary_kv,
        )

    raise ValueError(f"Unknown attn_type: {attn_type!r}")


def main(args):
    rank, _ = init_distributed()

    # Stop words from prompt config (same source as run_prelim_ruler.py)
    stop_words = PROMPT_TEMPLATES[args.prompt_config]["stop_words"] or None

    input_data = read_jsonl(args.input_path, args.num_samples)
    if not input_data:
        raise ValueError(f"No samples in {args.input_path}")

    model = load_model(
        model_path=args.model_path,
        attn_type=args.attn_type,
        tokens_to_generate=args.tokens_to_generate,
        stop_words=stop_words,
        block_size=args.block_size,
        anchor_block_size=args.anchor_block_size,
        summary_chunks=args.summary_chunks,
        chunk_size=args.chunk_size,
        summary_method=args.summary_method,
        sink_size=args.sink_size,
        discard_summary_kv=not args.no_discard_summary_kv,
    )

    warmup_sample = input_data[0]

    if rank == 0:
        print(f"Warmup: {args.warmup_rounds} round(s) on sample[0]...")
    for _ in range(args.warmup_rounds):
        model(
            prompt_context=warmup_sample["input_context"],
            prompt_query=warmup_sample["input_query"],
        )
        dist.barrier()

    if rank == 0:
        print(f"Timing {len(input_data)} samples...\n")

    import torch
    import torch.distributed as dist_mod

    def _peak_mem_gb_this_rank() -> float:
        """Peak allocated memory (GB) on this rank's primary device."""
        if not torch.cuda.is_available():
            return 0.0
        dev = torch.cuda.current_device()
        return torch.cuda.max_memory_allocated(dev) / 1024 ** 3

    def _all_devices_mem_gb() -> tuple:
        """For single-process (dense) mode: sum + max across ALL visible CUDA devices."""
        n = torch.cuda.device_count()
        per_dev = [torch.cuda.max_memory_allocated(i) / 1024 ** 3 for i in range(n)]
        return sum(per_dev), max(per_dev), per_dev

    # Reset peak stats before timed loop (post-warmup baseline)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sample_times: List[float] = []
    sample_peak_mem: List[float] = []   # peak per-GPU (this rank) per sample
    total_start = time.perf_counter()

    for i, sample in enumerate(input_data):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        model(
            prompt_context=sample["input_context"],
            prompt_query=sample["input_query"],
        )
        dist.barrier()
        elapsed = time.perf_counter() - t0
        peak_this = _peak_mem_gb_this_rank()

        if rank == 0:
            sample_times.append(elapsed)
            sample_peak_mem.append(peak_this)
            label = sample.get("index", i)
            print(f"  sample {i:>2} (idx={label}): {elapsed:.2f}s  peak={peak_this:.2f}GB")

    total_elapsed = time.perf_counter() - total_start

    # Gather per-rank peak memory at rank 0 for distributed runs
    # Each rank sends its max peak across all samples
    rank_max_peak = max(sample_peak_mem) if sample_peak_mem else 0.0
    if dist.is_initialized() and dist.get_world_size() > 1:
        peak_tensor = torch.tensor([rank_max_peak], dtype=torch.float32)
        all_peaks = [torch.zeros(1) for _ in range(dist.get_world_size())]
        dist.gather(peak_tensor, all_peaks if rank == 0 else None, dst=0)
        if rank == 0:
            per_rank_peaks = [t.item() for t in all_peaks]
        else:
            per_rank_peaks = []
    else:
        # Dense / single-process: enumerate all CUDA devices directly
        _, _, per_dev = _all_devices_mem_gb()
        per_rank_peaks = per_dev

    if rank == 0:
        mean_t = sum(sample_times) / len(sample_times)
        sorted_t = sorted(sample_times)
        mid = len(sorted_t) // 2
        median_t = (
            sorted_t[mid]
            if len(sorted_t) % 2 == 1
            else (sorted_t[mid - 1] + sorted_t[mid]) / 2
        )
        seq_len = os.path.basename(args.input_path).replace("timing_", "").replace(".jsonl", "")

        peak_per_gpu = max(per_rank_peaks) if per_rank_peaks else 0.0
        peak_total   = sum(per_rank_peaks) if per_rank_peaks else 0.0

        print()
        print("=" * 55)
        print(f"  attn_type      : {args.attn_type}")
        if args.attn_type == "star":
            print(f"  block_size     : {args.block_size}")
            print(f"  summary_method : {args.summary_method}")
            print(f"  summary_chunks : {args.summary_chunks}")
            print(f"  chunk_size     : {args.chunk_size}")
            print(f"  sink_size      : {args.sink_size}")
        print(f"  seq_len        : {seq_len}")
        print(f"  samples        : {len(sample_times)}")
        print(f"  total wall     : {total_elapsed:.1f}s")
        print(f"  mean / sample  : {mean_t:.2f}s")
        print(f"  median/ sample : {median_t:.2f}s")
        print(f"  min / max      : {min(sorted_t):.2f}s / {max(sorted_t):.2f}s")
        print(f"  peak / GPU     : {peak_per_gpu:.2f} GB  (max across ranks)")
        print(f"  peak total     : {peak_total:.2f} GB  (sum across ranks)")
        print(f"  per-rank peaks : {[f'{p:.2f}' for p in per_rank_peaks]} GB")
        print("=" * 55)

        if args.output_file:
            import datetime
            method_tag = (
                args.summary_method if args.attn_type == "star" else args.attn_type
            )
            line = (
                f"{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')} | "
                f"seq={seq_len} | "
                f"attn={args.attn_type} | "
                f"method={method_tag} | "
                f"block_size={args.block_size} | "
                f"summary_chunks={args.summary_chunks} | "
                f"chunk_size={args.chunk_size} | "
                f"sink_size={args.sink_size} | "
                f"n={len(sample_times)} | "
                f"mean={mean_t:.3f}s | "
                f"median={median_t:.3f}s | "
                f"min={min(sorted_t):.3f}s | "
                f"max={max(sorted_t):.3f}s | "
                f"total={total_elapsed:.1f}s | "
                f"peak_per_gpu={peak_per_gpu:.3f}GB | "
                f"peak_total={peak_total:.3f}GB\n"
            )
            os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
            with open(args.output_file, "a", encoding="utf-8") as f:
                f.write(line)
            print(f"\n  Results appended to: {args.output_file}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timing harness for dense / star-anchor / statistical inference."
    )

    # Model
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--prompt_config",
        required=True,
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Prompt template — determines stop_words (same as run_prelim_ruler.py).",
    )
    parser.add_argument("--attn_type", required=True, choices=["dense", "ring", "star"])
    parser.add_argument("--tokens_to_generate", type=int, default=128)

    # Star Attention / Statistical Cross-Attention  (matches run_prelim_ruler.py)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument(
        "--anchor_block_size",
        type=int,
        default=-1,
        help="-1 = no anchor (use summaries only). >0 = anchor mode.",
    )
    parser.add_argument(
        "--summary_method",
        default="tfidf",
        choices=["tfidf", "bm25", "entropy", "max_idf", "evenly_spaced", "mean_pool"],
    )
    parser.add_argument("--summary_chunks", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--no_discard_summary_kv", action="store_true")

    # Data
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Max samples to time (-1 = all).",
    )

    # Warmup
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=3,
        help="Repeat sample[0] this many times before timing starts.",
    )

    # Output
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to append one structured result line per run (e.g. results/timing.txt).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input not found: {args.input_path}")

    main(args)

#!/usr/bin/env bash
# run_timing_benchmark.sh — Full timing sweep for dense / anchor / max_idf
#
# Usage:
#   bash scripts/run_timing_benchmark.sh <model_path> [output_dir]
#
# Example:
#   bash scripts/run_timing_benchmark.sh meta-llama/Llama-3.1-8B-Instruct results/timing
#
# Runs 12 timed experiments (3 modes x 4 seq lengths) and writes:
#   <output_dir>/timing_results.txt   — one structured line per run (append)
#   <output_dir>/logs/<run_tag>.log   — full stdout+stderr per run
#
# Sequence lengths : 16K, 32K, 64K, 128K
# Block counts     : 4 blocks for 16/32/64K  |  8 blocks for 128K
# Summary budget   : 12.5% of block_size, chunk_size=32
# Hardware         : 4x A100-40GB
#   dense  -> torchrun --nproc_per_node=1  (device_map=auto across 4 GPUs)
#   star   -> torchrun --nproc_per_node=4  (1 GPU per rank; 2 blocks/rank @ 128K)

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
MODEL=${1:?"Usage: $0 <model_path> [output_dir]"}
OUTPUT_DIR=${2:-"results/timing"}
RESULTS_FILE="$OUTPUT_DIR/timing_results.txt"
LOG_DIR="$OUTPUT_DIR/logs"
DATA_DIR="dataset/timing"
PROMPT_CFG="llama3"
WARMUP=3

mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

run_timed() {
    local tag="$1"; shift
    local log="$LOG_DIR/${tag}.log"
    log "START  $tag"
    if torchrun "$@" 2>&1 | tee "$log"; then
        log "DONE   $tag  (log: $log)"
    else
        log "FAILED $tag — see $log"
        exit 1
    fi
}

# ── Step 1: Generate datasets ─────────────────────────────────────────────────
log "=== Generating RULER datasets ==="
TASKS=(
    niah_single_1 niah_single_2 niah_single_3
    niah_multikey_1 niah_multikey_2 niah_multikey_3
    niah_multivalue niah_multiquery
    vt cwe fwe qa_1 qa_2
)

for SEQ in 16384 32768 65536 131072; do
    MERGED="$DATA_DIR/timing_${SEQ}.jsonl"
    if [[ -f "$MERGED" ]]; then
        log "  timing_${SEQ}.jsonl exists — skipping data gen"
        continue
    fi
    log "  Generating seq_length=$SEQ ..."
    for TASK in "${TASKS[@]}"; do
        TASK_FILE="$DATA_DIR/$SEQ/$TASK/validation.jsonl"
        if [[ ! -f "$TASK_FILE" ]]; then
            python ruler/data/prepare.py \
                --save_dir "$DATA_DIR/$SEQ" \
                --task "$TASK" \
                --tokenizer_path "$MODEL" \
                --tokenizer_type hf \
                --max_seq_length "$SEQ" \
                --model_template_type "$PROMPT_CFG" \
                --num_samples 1
        fi
    done
    # Merge 13 task files into one jsonl
    cat "$DATA_DIR/$SEQ"/*/validation.jsonl > "$MERGED"
    log "  Merged -> $MERGED ($(wc -l < "$MERGED") lines)"
done

# ── Step 2: Timing runs ────────────────────────────────────────────────────────
log "=== Starting timing runs ==="
log "    Results -> $RESULTS_FILE"

BASE_ARGS=(
    --prompt_config "$PROMPT_CFG"
    --tokens_to_generate 128
    --warmup_rounds "$WARMUP"
    --output_file "$RESULTS_FILE"
)

# ─────────────────────────── 16K — 4 blocks — block=4096 ───────────────────
INPUT="$DATA_DIR/timing_16384.jsonl"

run_timed "16k_dense" \
    --nproc_per_node=1 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type dense \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "16k_anchor" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 4096 --summary_method anchor \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "16k_maxidf" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 4096 --summary_method max_idf \
    --summary_chunks 16 --chunk_size 32 --sink_size 64 \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

# ─────────────────────────── 32K — 4 blocks — block=8192 ───────────────────
INPUT="$DATA_DIR/timing_32768.jsonl"

run_timed "32k_dense" \
    --nproc_per_node=1 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type dense \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "32k_anchor" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 8192 --summary_method anchor \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "32k_maxidf" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 8192 --summary_method max_idf \
    --summary_chunks 32 --chunk_size 32 --sink_size 64 \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

# ─────────────────────────── 64K — 4 blocks — block=16384 ──────────────────
INPUT="$DATA_DIR/timing_65536.jsonl"

run_timed "64k_dense" \
    --nproc_per_node=1 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type dense \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "64k_anchor" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 16384 --summary_method anchor \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "64k_maxidf" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 16384 --summary_method max_idf \
    --summary_chunks 64 --chunk_size 32 --sink_size 64 \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

# ─────────────────────────── 128K — 8 blocks — block=16384 — 2 blk/GPU ────
INPUT="$DATA_DIR/timing_131072.jsonl"

run_timed "128k_dense" \
    --nproc_per_node=1 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type dense \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "128k_anchor" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 16384 --summary_method anchor \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

run_timed "128k_maxidf" \
    --nproc_per_node=4 scripts/time_inference.py \
    --model_path "$MODEL" --attn_type star \
    --block_size 16384 --summary_method max_idf \
    --summary_chunks 64 --chunk_size 32 --sink_size 64 \
    --input_path "$INPUT" "${BASE_ARGS[@]}"

# ── Done ──────────────────────────────────────────────────────────────────────
log "=== All runs complete ==="
log "    Results file : $RESULTS_FILE"
log "    Run logs     : $LOG_DIR/"
echo ""
echo "Results:"
cat "$RESULTS_FILE"

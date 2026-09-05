#!/usr/bin/env bash
# Orchestrates the remaining reruns for the HRI paper once robot-manual
# ingestion has finished: ingest the small DocBench subsample, train the
# learned-alpha model, rerun sensitivity analysis, and run the quantitative
# adversarial ASR study. Each step logs to its own file under /tmp so
# progress can be checked independently.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

run_step() {
  local name="$1" logfile="$2"; shift 2
  echo "=== $name ==="
  "$@" > "$logfile" 2>&1
  local rc=$?
  echo "  exit code $rc, see $logfile"
  return 0  # never abort the overall script on a single step's failure
}

run_step "[1/4] Ingesting small DocBench subsample (6 docs, ~63 pages)" /tmp/ingest_docbench_sample.log \
  python main.py --mode ingest --files "data/raw/DocBenchSample/0/*.pdf" "data/raw/DocBenchSample/1/*.pdf" "data/raw/DocBenchSample/10/*.pdf" "data/raw/DocBenchSample/11/*.pdf" "data/raw/DocBenchSample/100/*.pdf" "data/raw/DocBenchSample/103/*.pdf"

run_step "[2/4] Training learned-alpha model" /tmp/train_learned_alpha.log \
  python scripts/train_learned_alpha.py --doc_bench_dir data/raw/DocBenchSample --top_k 10

run_step "[3/4] Running sensitivity analysis (beta sweep)" /tmp/sensitivity_run.log \
  python evaluate.py --mode sensitivity --doc_bench_dir data/raw/DocBenchSample --top_k 10

run_step "[4/4] Running quantitative adversarial ASR study" /tmp/asr_run.log \
  python -m src.evaluators.adversarial_asr_evaluator --doc_bench_dir data/raw/DocBenchSample --sample_size 20 --top_k 10

echo "=== All remaining reruns attempted -- check each log above for actual success/failure ==="

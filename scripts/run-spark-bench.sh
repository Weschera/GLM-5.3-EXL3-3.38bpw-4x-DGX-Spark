#!/usr/bin/env bash
# run-spark-bench.sh - reproduce our Spark-Bench v7.0 run against the running server.
# Spark-Bench: https://github.com/Weschera/spark-bench (we ran grader commit 509759f).
# Run where the bench lives; reach rank 0's loopback /v1 with:  ssh -N -L 18890:127.0.0.1:8890 <user>@<rank-0 Spark>
set -euo pipefail
ENDPOINT="${ENDPOINT:-http://127.0.0.1:18890/v1}"
OUT="${OUT:-./g338-off}"
export SPARK_BENCH_SEED_PER_REPEAT=2000 PYTHONUNBUFFERED=1 SPARK_BENCH_FORCE_TEMPERATURE=1.0
export SPARK_BENCH_CONTEXT=163840 SPARK_BENCH_EXTRA_BODY='{"top_p":0.95}'
python spark_bench.py eval \
  --label "GLM-5.3 EXL3 SAGE MixedK 3.38bpw (TensorFold TP4 + DFlash2, 4 Sparks, fast-160k) thinking OFF" \
  --endpoint "$ENDPOINT" --model GLM-5.3-EXL3-3.38bpw --topology "4x DGX Spark TP4 (RoCE)" \
  --thinking off --uncapped --temperature 1.0 --repeats 2 --tier all --timeout 7200 --preflight-loose \
  --out-dir "$OUT" --run-kind benchmark

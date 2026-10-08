#!/usr/bin/env bash
# prebuild-extensions.sh - build TensorFold's three CUDA extensions on every Spark BEFORE ./glm53 up.
#
# Why: on first start each rank JIT-compiles tensorfold_qmm_v5, tensorfold_exl3_linear_v5 and
# tensorfold_exl3_experts_v2 (nvcc + ninja) while the 3.38bpw weights are already resident. On our cluster that left
# one Spark at ~6 GiB free; the compile pushed it under 2 GiB and the recipe's memory guard stopped all four ranks.
# Building them first, with nothing loaded, takes a few minutes and later starts reuse the .so files.
#
# Run on rank 0 from the recipe clone, after `./glm53 setup` and before `./glm53 up`:
#   bash /path/to/prebuild-extensions.sh
set -euo pipefail
RECIPE="${RECIPE:-$HOME/GLM-5.3-EXL3-DGX-Spark-recipe}"
HOSTS="$RECIPE/tensorfold-four-spark-tp4/hosts"
VENV="${VENV:-$HOME/glm53-tensorfold/venv}"
[[ -f "$HOSTS" ]] || { echo "no hosts file at $HOSTS (run ./glm53 init or write it by hand)"; exit 1; }

build='set -e
for d in "$HOME/.cache/torch_extensions" "$HOME/.triton"; do
  [[ -e "$d" ]] && [[ -n "$(find "$d" ! -user "$(id -un)" -print -quit 2>/dev/null)" ]] && {
    echo "$d has files owned by another user (an earlier root run). Fix: sudo chown -R $(id -un):$(id -gn) $d"; exit 1; }
done
# a killed earlier build leaves a lock that makes every later start wait forever; no build runs now, so clear them
find "$HOME"/.cache/torch_extensions/py*/tensorfold_* -maxdepth 1 -name lock -delete 2>/dev/null || true
cd "$HOME/glm53-tensorfold"
"'"$VENV"'/bin/python" - <<PY
import time; t = time.time()
from tensorfold.cuda.kernels import qmm; qmm._ext()
from tensorfold.cuda.exl3 import linear, experts; linear._ext(); experts._ext()
print("extensions ready in %.0f s" % (time.time() - t))
PY
ls "$HOME"/.cache/torch_extensions/py*/tensorfold_*/*.so'

pids=()
while read -r rank target _; do
  [[ "$rank" =~ ^[0-3]$ ]] || continue
  if [[ "$target" == local ]]; then
    ( bash -c "$build" 2>&1 | sed "s/^/rank $rank: /" ) & pids+=($!)
  else
    ( ssh -o BatchMode=yes "$target" "bash -s" <<<"$build" 2>&1 | sed "s/^/rank $rank: /" ) & pids+=($!)
  fi
done < "$HOSTS"
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail == 0 ]] && echo "prebuild OK on all four. Next: ./glm53 up" || { echo "prebuild FAILED on at least one rank"; exit 1; }

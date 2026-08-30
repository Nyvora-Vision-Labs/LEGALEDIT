#!/bin/bash
# Drive the exp3 grid one process per (config, seed).
#
# Three things this script exists to handle, all learned the hard way:
#
#  1. MPS does not reliably release memory between runs inside one process. The
#     second seed of a config ran 2.2x slower than the first and the third died
#     with an out-of-memory error. Process isolation gives each run a clean
#     allocator.
#  2. macOS puts processes spawned from a detached background shell into the
#     background QoS class, which throttles MPS work roughly 4x (one run took
#     44 min of wall clock against a 9.5 min baseline). `taskpolicy -B -p` on
#     the child, right after it starts, removes that clamp.
#  3. The grid takes hours and has been interrupted repeatedly, so it skips any
#     (config, seed) already present in the results JSON and can be re-run at
#     any point to fill in what is missing.
set -u
cd "$(dirname "$0")"

TAG="${TAG:-}"
GROUPED="${GROUPED:-}"
SEEDS="${SEEDS:-42 43 44 45 46}"
CONFIGS="${CONFIGS:-JudgeBERT (repro)|JudgeBERT-Dist (soft labels)|JudgeBERT-Annot (5 heads)|JudgeBERT-MT (+charact.)|JudgeBERT-Quantile (tau=.25)|JudgeBERT-DA (repro)|JudgeBERT-DA+LegalEdit}"

JSON="../results/exp3_train${TAG}.json"
LOG="../results/exp3_run${TAG}.log"
echo "=== grid started $(date) tag='${TAG}' grouped='${GROUPED}' ===" >> "$LOG"

done_already() {  # $1 = config, $2 = seed
  python3 - "$JSON" "$1" "$2" <<'PY'
import json, os, sys
f, cfg, sd = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not os.path.exists(f):
    sys.exit(1)
d = json.load(open(f))
sys.exit(0 if any(r["seed"] == sd for r in d.get(cfg, [])) else 1)
PY
}

IFS='|' read -ra CFGS <<< "$CONFIGS"
for cfg in "${CFGS[@]}"; do
  for sd in $SEEDS; do
    if done_already "$cfg" "$sd"; then
      echo "--- skip (done) $cfg seed=$sd" >> "$LOG"; continue
    fi
    echo "--- $cfg seed=$sd $(date +%H:%M:%S)" >> "$LOG"
    python3 -u exp3_train.py --configs "$cfg" --seeds "$sd" \
        ${GROUPED:+--grouped} ${TAG:+--tag "$TAG"} >> "$LOG" 2>&1 &
    child=$!
    sleep 2; taskpolicy -B -p $child 2>/dev/null
    wait $child
    rc=$?
    [ $rc -ne 0 ] && echo "!!! $cfg seed=$sd exited $rc" >> "$LOG"
  done
done
echo "=== grid finished $(date) ===" >> "$LOG"

#!/usr/bin/env bash
# Wait for a SuperMimic-Bumi training unit to finish, then evaluate all its checkpoints with N parallel
# eval processes.
#
#   PYTHON=<python of the Isaac Lab env> supermimic/eval_after_train.sh <run> [N=3]
#   -> logs/<run>/eval.csv, ckpt_*_eval.json (per-clip), eval_shard*.out
set -uo pipefail
RUN=${1:?run name}; N=${2:-3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=$REPO/logs/$RUN
PY=${PYTHON:-$(command -v python)}
UNIT="supermimic-bumi-$RUN-train"
while systemctl --user is-active -q "$UNIT"; do sleep 60; done
echo "$(date '+%F %T') training unit inactive; evaluating $(ls "$LOG"/ckpt_*.pt | wc -l) checkpoints with $N shards"
cd "$REPO"
[ -f "$LOG/eval.csv" ] || echo "iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed" > "$LOG/eval.csv"
pids=()
for ((i = 0; i < N; i++)); do
  "$PY" supermimic/evaluate.py --headless --run_dir "$LOG" \
    --shard "$i/$N" > "$LOG/eval_shard$i.out" 2>&1 &
  pids+=($!)
  sleep 60  # stagger: concurrent starts race on the shared URDF->USD cache
done
wait "${pids[@]}"
echo "$(date '+%F %T') done"
(head -1 "$LOG/eval.csv"; tail -n +2 "$LOG/eval.csv" | sort -t, -k1,1n) | cut -d, -f1,2,5,6,7 | column -t -s,

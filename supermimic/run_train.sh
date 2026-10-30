#!/usr/bin/env bash
# Launch SuperMimic-Bumi training as a systemd user unit so it outlives the shell.
#
#   PYTHON=<python of the Isaac Lab env> supermimic/run_train.sh <run_name> [extra train.py args...]
#   systemctl --user status supermimic-bumi-<run_name>-train ; tail -f logs/<run_name>/train.out
#
# 4096 envs / 1000-step buffer / --grad_ckpt fit a 16 GB GPU, with no room left for an eval watcher: evaluate after
# training (eval_after_train.sh).
set -euo pipefail
RUN=${1:?run name}; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=$REPO/logs/$RUN
mkdir -p "$LOG"
PY=${PYTHON:-$(command -v python)}

systemd-run --user --unit="supermimic-bumi-$RUN-train" --collect -p WorkingDirectory="$REPO" \
  bash -c "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec $PY supermimic/train.py --headless --num_envs ${NUM_ENVS:-4096} --buffer_steps ${BUFFER:-1000} --grad_ckpt \
    --log_dir $LOG $* >> $LOG/train.out 2>&1"
echo "log dir: $LOG"

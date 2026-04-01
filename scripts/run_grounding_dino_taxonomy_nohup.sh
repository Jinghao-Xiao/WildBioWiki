#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/jinxiao/jinxiao/wildlife_vqa"
OUTPUT_ROOT="$PROJECT_ROOT/data/iNaturalist/grounding_dino_boxes"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

TAXONOMY_CLASS="${1:-Actinopterygii}"
GPU_INDEX="${2:-1}"
MODE="${3:-start}"

PID_FILE="$OUTPUT_ROOT/${TAXONOMY_CLASS}.gpu${GPU_INDEX}.pid"
LATEST_LOG_LINK="$OUTPUT_ROOT/${TAXONOMY_CLASS}.gpu${GPU_INDEX}.latest.log"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

case "$MODE" in
  status)
    if is_running; then
      echo "Running PID $(cat "$PID_FILE")"
      echo "Log $(readlink -f "$LATEST_LOG_LINK")"
    else
      echo "Not running"
      [[ -f "$PID_FILE" ]] && echo "Stale PID file: $PID_FILE"
    fi
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PID_FILE")"
      echo "Stopped PID $(cat "$PID_FILE")"
    else
      echo "Not running"
    fi
    ;;
  start)
    if is_running; then
      echo "Already running PID $(cat "$PID_FILE")"
      exit 0
    fi

    LOG_FILE="$LOG_DIR/${TAXONOMY_CLASS}_gpu${GPU_INDEX}_$(date +%Y%m%d_%H%M%S).log"
    ln -sfn "$LOG_FILE" "$LATEST_LOG_LINK"

    nohup setsid bash -lc "
      source /data/jinxiao/miniconda3/etc/profile.d/conda.sh
      conda activate wildlife_vqa
      cd '$PROJECT_ROOT'
      export CUDA_VISIBLE_DEVICES='$GPU_INDEX'
      exec python -u scripts/run_grounding_dino_taxonomy.py \
        --taxonomy-class '$TAXONOMY_CLASS' \
        --device cuda:0 \
        --image-root data/iNaturalist/inaturalist_images_396 \
        --prompt-csv data/iNaturalist/inat_detector_prompts_396.csv \
        --model-dir models/grounding-dino-base \
        --output-root data/iNaturalist/grounding_dino_boxes
    " >"$LOG_FILE" 2>&1 </dev/null &

    echo "$!" >"$PID_FILE"
    echo "Started PID $(cat "$PID_FILE")"
    echo "Log $LOG_FILE"
    ;;
  *)
    echo "Usage: $0 [taxonomy_class] [gpu_index] [start|status|stop]"
    exit 1
    ;;
esac

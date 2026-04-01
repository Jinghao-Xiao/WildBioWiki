#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/jinxiao/Data/jinxiao/wildlife_vqa"
OUTPUT_ROOT="$PROJECT_ROOT/data/WildBioWiki-QA-medium"
LOG_DIR="$OUTPUT_ROOT/logs"
PID_FILE="$OUTPUT_ROOT/run.pid"

mkdir -p "$LOG_DIR"

source /data/jinxiao/miniconda3/etc/profile.d/conda.sh
conda activate wildlife_vqa

action="${1:-start}"

case "$action" in
  start)
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="$LOG_DIR/build_medium_${timestamp}.log"
    nohup python "$PROJECT_ROOT/scripts/build_wildbiowiki_qa_medium.py" \
      --input-root data/WildBioWiki-QA \
      --metadata-root data/iNaturalist/inaturalist_images_396 \
      --output-root data/WildBioWiki-QA-medium \
      --workers 16 \
      --timeout 30 \
      --max-retries 5 \
      > "$log_file" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "started pid=$(cat "$PID_FILE") log=$log_file"
    ;;
  status)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "running pid=$pid"
      else
        echo "stale pid file pid=$pid"
      fi
    else
      echo "not running"
    fi
    ;;
  stop)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      if ps -p "$pid" > /dev/null 2>&1; then
        kill "$pid"
        echo "stopped pid=$pid"
      else
        echo "stale pid file pid=$pid"
      fi
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 {start|status|stop}" >&2
    exit 1
    ;;
esac

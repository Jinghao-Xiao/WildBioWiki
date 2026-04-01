#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/data/iNaturalist/inaturalist_images_396"
LOG_DIR="$OUTPUT_ROOT/logs"
PID_FILE="$OUTPUT_ROOT/run.pid"

MODE="${1:-start}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

mkdir -p "$LOG_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$MODE" in
  start)
    if is_running; then
      echo "Download job is already running with PID $(cat "$PID_FILE")"
      exit 0
    fi

    source /data/jinxiao/miniconda3/etc/profile.d/conda.sh
    conda activate wildlife_vqa

    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/download_${TIMESTAMP}.log"

    nohup setsid python -u "$ROOT_DIR/scripts/download_inaturalist_images.py" \
      --manifest inat_download_manifest_396.jsonl \
      --output-root data/iNaturalist/inaturalist_images_396 \
      --target-per-species 800 \
      --api-per-page 200 \
      --download-workers 16 \
      --request-timeout 30 \
      --max-retries 5 \
      --resume \
      "$@" \
      >"$LOG_FILE" 2>&1 < /dev/null &

    echo $! > "$PID_FILE"
    echo "Started download job"
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: $LOG_FILE"
    ;;

  status)
    if is_running; then
      echo "Running"
      echo "PID: $(cat "$PID_FILE")"
      ps -fp "$(cat "$PID_FILE")"
    else
      echo "Not running"
      if [[ -f "$PID_FILE" ]]; then
        echo "Stale PID file: $(cat "$PID_FILE")"
      fi
    fi
    ;;

  stop)
    if is_running; then
      kill "$(cat "$PID_FILE")"
      echo "Stopped PID $(cat "$PID_FILE")"
      rm -f "$PID_FILE"
    else
      echo "No running job found"
      rm -f "$PID_FILE"
    fi
    ;;

  *)
    echo "Usage: $0 [start|status|stop] [extra downloader args...]"
    exit 1
    ;;
esac

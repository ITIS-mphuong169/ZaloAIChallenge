#!/usr/bin/env bash
# Entry point for the organisers' run: test set mounted read-only at /data, results in /result.
set -euo pipefail
DATA_DIR="${DATA_DIR:-/data}"
RESULT_DIR="${RESULT_DIR:-/result}"
mkdir -p "${RESULT_DIR}/logs"
python -m datamate.cli predict --data "${DATA_DIR}" --out "${RESULT_DIR}/submission.jsonl" --log "${RESULT_DIR}/logs/run_log.jsonl"

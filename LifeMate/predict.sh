#!/usr/bin/env bash
# Entry point of the LifeMate image. The test set is mounted read-only at /data, results are written to /result.
#
#   1. speech -> text for every clip of the split   (the GPU is free again when this step ends)
#   2. start the language-model server and wait until it answers
#   3. run the agent and write /result/submission.jsonl
#
# A failing step never stops the run: the next steps go on with what exists, so a submission file is always
# written (a sample that could not be handled gets the "nothing done" answer).
# Every setting below can be overridden with an environment variable; that is how the script is tested outside Docker.
set -uo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RESULT_DIR="${RESULT_DIR:-/result}"
PYTHON="${PYTHON:-python3}"
ASR_MODEL="${ASR_MODEL:-/models/asr}"
ASR_ARGS="${ASR_ARGS:---backend hf}"
LLM_MODEL="${LLM_MODEL:-/models/llm}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_ARGS="${LLM_ARGS:---max-model-len 16384 --gpu-memory-utilization 0.92}"
LLM_WAIT="${LLM_WAIT:-1800}"            # seconds allowed for the server to load the model
PREDICT_ARGS="${PREDICT_ARGS:---max-tokens 2048 --workers 8}"
SAMPLE_ARGS="${SAMPLE_ARGS:-}"          # for a quick test, e.g. "--limit 5"; empty = the whole split
export LIFEMATE_TRAIN_JSONL="${LIFEMATE_TRAIN_JSONL:-/app/train/train.jsonl}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

log() { echo "[lifemate $(date +%H:%M:%S)] $*"; }
mkdir -p "$RESULT_DIR/logs"

# ---- find the split.  Layout A: $DATA_DIR/<split>/<split>.jsonl    Layout B: $DATA_DIR/<split>.jsonl
find_split() {
  local f d n
  if [ -n "${SPLIT:-}" ]; then
    [ -f "$DATA_DIR/$SPLIT/$SPLIT.jsonl" ] && { echo "A $SPLIT"; return 0; }
    [ -f "$DATA_DIR/$SPLIT.jsonl" ] && { echo "B $SPLIT"; return 0; }
    return 1
  fi
  for f in "$DATA_DIR"/*/*.jsonl; do
    [ -f "$f" ] || continue
    d="$(dirname "$f")"; n="$(basename "$d")"
    if [ "$(basename "$f")" = "$n.jsonl" ] && [ "$n" != "train" ] && [ -f "$d/${n}_tools_catalog.json" ]; then echo "A $n"; return 0; fi
  done
  for f in "$DATA_DIR"/*_tools_catalog.json; do
    [ -f "$f" ] || continue
    n="$(basename "$f" _tools_catalog.json)"
    if [ -f "$DATA_DIR/$n.jsonl" ]; then echo "B $n"; return 0; fi
  done
  return 1
}

FOUND="$(find_split)" || { log "no split found under $DATA_DIR (expected <split>/<split>.jsonl or <split>.jsonl)"; ls -la "$DATA_DIR" || true; exit 2; }
LAYOUT="${FOUND%% *}"
SPLIT="${FOUND#* }"
if [ "$LAYOUT" = "B" ]; then  # the code expects a folder named after the split: give it one through a link
  LINK_ROOT="$(mktemp -d)"
  ln -s "$(cd "$DATA_DIR" && pwd)" "$LINK_ROOT/$SPLIT"
  export LIFEMATE_DATA="$LINK_ROOT"
else
  export LIFEMATE_DATA="$DATA_DIR"
fi
log "split '$SPLIT' (layout $LAYOUT) under $DATA_DIR, results in $RESULT_DIR"

# ---- 1. speech -> text
log "1/3 speech to text with $ASR_MODEL"
# shellcheck disable=SC2086
"$PYTHON" -m lifemate.cli asr --split "$SPLIT" --model "$ASR_MODEL" $ASR_ARGS $SAMPLE_ARGS \
    --cache "$RESULT_DIR/logs/asr.jsonl" 2>&1 | tee "$RESULT_DIR/logs/asr.log" | grep -v -E '^\[asr\] [0-9]+/' | tail -n 5
code="${PIPESTATUS[0]}"
[ "$code" -eq 0 ] || log "speech step ended with code $code; going on with the clips that were transcribed"

# ---- 2. language-model server
log "2/3 starting the language-model server"
if [ -n "${LLM_SERVE_CMD:-}" ]; then
  bash -c "$LLM_SERVE_CMD" > "$RESULT_DIR/logs/llm_server.log" 2>&1 &
else
  # shellcheck disable=SC2086
  vllm serve "$LLM_MODEL" --served-model-name llm --port "$LLM_PORT" $LLM_ARGS > "$RESULT_DIR/logs/llm_server.log" 2>&1 &
fi
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT

"$PYTHON" - "$LLM_PORT" "$LLM_WAIT" "$SERVER_PID" <<'PY'
import sys
import time
import urllib.request

port, wait, pid = sys.argv[1], int(sys.argv[2]), sys.argv[3]


def alive() -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


t0 = time.time()
while time.time() - t0 < wait:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read()
        print(f"server ready after {time.time() - t0:.0f}s")
        sys.exit(0)
    except Exception:
        pass
    if not alive():
        print("server process exited before it was ready")
        sys.exit(1)
    time.sleep(3)
print(f"server not ready after {wait}s")
sys.exit(1)
PY
if [ $? -ne 0 ]; then
  log "the language-model server is not available; last lines of its log:"
  tail -n 20 "$RESULT_DIR/logs/llm_server.log" || true
  log "going on: samples will get the 'nothing done' answer"
fi

# ---- 3. agent
log "3/3 running the agent"
# shellcheck disable=SC2086
"$PYTHON" -m lifemate.cli predict --split "$SPLIT" --asr "$RESULT_DIR/logs/asr.jsonl" --allow-missing \
    --base-url "http://127.0.0.1:$LLM_PORT/v1" --model llm $PREDICT_ARGS $SAMPLE_ARGS \
    --out "$RESULT_DIR/submission.jsonl" --trace "$RESULT_DIR/logs/trace.jsonl" --fresh 2>&1 | tee "$RESULT_DIR/logs/predict.log" | tail -n 15
code="${PIPESTATUS[0]}"

if [ -s "$RESULT_DIR/submission.jsonl" ]; then
  log "done: $(wc -l < "$RESULT_DIR/submission.jsonl") lines in $RESULT_DIR/submission.jsonl (agent exit code $code)"
  exit 0
fi
log "FAILED: no submission file was written (agent exit code $code)"
exit 1

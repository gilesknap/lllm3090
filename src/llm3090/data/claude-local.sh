#!/bin/bash
# Launch Claude Code against whatever is serving on 127.0.0.1:1919.
#
# Works for both engines: FreeToken and llama.cpp each expose Anthropic's
# /v1/messages, so Claude Code cannot tell the difference. Nothing is written to
# ~/.claude/settings.json — these are per-process environment variables only, so a
# plain `claude` in another terminal still reaches Anthropic on your normal auth.
#
# Usage:  ./claude-local.sh [any further claude args]
set -euo pipefail

ENDPOINT="http://127.0.0.1:1919"

if ! curl -sf --max-time 3 "$ENDPOINT/health" >/dev/null 2>&1 &&
   ! curl -sf --max-time 3 "$ENDPOINT/v1/models" >/dev/null 2>&1; then
  echo "Nothing is serving on $ENDPOINT." >&2
  echo "Start a model at http://127.0.0.1:8080 first." >&2
  exit 1
fi

# Ask the server what it is actually serving rather than hardcoding a name --
# the two engines report it under different keys.
MODEL="$(curl -sf --max-time 5 "$ENDPOINT/v1/models" \
  | python3 -c 'import json,sys
d=json.load(sys.stdin)
m=(d.get("data") or d.get("models") or [{}])[0]
print(m.get("id") or m.get("name") or "")' 2>/dev/null || true)"

if [ -z "$MODEL" ]; then
  echo "Could not determine the served model from $ENDPOINT/v1/models" >&2
  exit 1
fi

# Context: FreeToken reports it; llama.cpp does not, so fall back to the panel's figure.
CTX="$(curl -sf --max-time 5 "http://127.0.0.1:8080/api/status" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("context_length") or "")' \
  2>/dev/null || true)"
CTX="${CTX:-131072}"

echo "Claude Code -> $MODEL @ $ENDPOINT (context ${CTX})"

exec env \
  ANTHROPIC_BASE_URL="$ENDPOINT" \
  ANTHROPIC_AUTH_TOKEN="local" \
  ANTHROPIC_API_KEY="" \
  ANTHROPIC_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
  CLAUDE_CODE_SUBAGENT_MODEL="$MODEL" \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CTX" \
  CLAUDE_CODE_MAX_OUTPUT_TOKENS="32768" \
  claude "$@"

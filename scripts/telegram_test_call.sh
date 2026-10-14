#!/usr/bin/env bash
set -euo pipefail

# ── Telegram test call ────────────────────────────────────────────────
# Places a real voice call from the logged-in Telegram user account and
# prints the outcome (transcript on success, the exact error on failure).
#
# Usage:
#   ./scripts/telegram_test_call.sh [@username] [max_minutes] [greeting]
#
# Defaults: @MentalCr1sis, 2 minutes, a short greeting.
# Stop `jarvis telegram listen` first: only one process may use the session.
# ──────────────────────────────────────────────────────────────────────

TARGET="${1:-@MentalCr1sis}"
MAX_MINUTES="${2:-2}"
GREETING="${3:-Hi, this is Jarvis. Can you hear me?}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# freellmapi key/URLs for the LLM, speech-to-text and text-to-speech.
ENV_FILE="${OPENJARVIS_HOME:-$HOME/.openjarvis}/freellmapi.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

echo "Calling $TARGET for up to $MAX_MINUTES minute(s)..."
uv run python - "$TARGET" "$MAX_MINUTES" "$GREETING" <<'PY'
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
# ntgcalls logs every relay ping as a WARNING; keep real errors only.
for noisy, level in (("telethon", logging.WARNING), ("pytgcalls", logging.WARNING),
                     ("ntgcalls", logging.ERROR), ("httpx", logging.WARNING)):
    logging.getLogger(noisy).setLevel(level)

from openjarvis.connectors.telegram_user import TelegramUserClient
from openjarvis.tools.telegram_user_tools import TelegramCallStartTool

target, max_minutes, greeting = sys.argv[1], float(sys.argv[2]), sys.argv[3]
result = TelegramCallStartTool().execute(
    chat=target, greeting=greeting, max_minutes=max_minutes, wait=True
)
TelegramUserClient.get().shutdown()

if not result.success:
    print(f"\nCALL FAILED: {result.content}")
    sys.exit(1)

call = json.loads(result.content)
print(f"\nCall with {call['title']} ended after {call['duration_s']}s "
      f"({call['end_reason'] or 'still active'}).")
for turn in call["transcript"]:
    print(f"  {turn['speaker']}: {turn['content']}")
PY

#!/bin/bash
set -euo pipefail
# ============================================================
# Load Groq API keys from 1Password → claude-mem settings
#
# Usage: ./scripts/load-groq-keys.sh
#
# Updates ~/.claude-mem/settings.json with:
#   CLAUDE_MEM_GROQ_API_KEY  (first available key)
#   CLAUDE_MEM_GROQ_API_KEYS (remaining keys, comma-separated)
#
# Keys are resolved via op-connect-load.sh (Connect API or op CLI).
# Pattern follows apps/proxy/ccproxy/source/scripts/load-groq-keys.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLAUDE_PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

# Self-load secrets from 1Password
source "$CLAUDE_PROJECT_DIR/scripts/op-connect-load.sh"

SETTINGS_FILE="$HOME/.claude-mem/settings.json"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Settings file not found: $SETTINGS_FILE"
    exit 1
fi

# Map env var names to indices (matches 1Password items — no underscore before number)
declare -A KEY_MAP=(
    [1]="GROQ_API_KEY"
    [2]="GROQ_API_KEY2"
    [3]="GROQ_API_KEY3"
    [4]="GROQ_API_KEY4"
    [5]="GROQ_API_KEY5"
)

keys=()
skipped=0

for idx in 1 2 3 4 5; do
    env_var="${KEY_MAP[$idx]}"
    value="${!env_var:-}"

    if [ -z "$value" ]; then
        echo "  $env_var not set — skipping"
        skipped=$((skipped + 1))
        continue
    fi

    keys+=("$value")
    echo "  key $idx <- $env_var (${value:0:12}...${value: -6})"
done

if [ ${#keys[@]} -eq 0 ]; then
    echo "No Groq keys found in environment"
    exit 1
fi

# First key = primary, rest = pool
primary="${keys[0]}"
pool=""
if [ ${#keys[@]} -gt 1 ]; then
    pool=$(IFS=,; echo "${keys[*]:1}")
fi

# Update settings.json via python (safe JSON handling)
python3 -c "
import json

with open('$SETTINGS_FILE') as f:
    settings = json.load(f)

settings['CLAUDE_MEM_GROQ_API_KEY'] = '$primary'
settings['CLAUDE_MEM_GROQ_API_KEYS'] = '$pool'

with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
"

echo ""
echo "Done: ${#keys[@]} keys loaded, $skipped skipped"
echo "Primary: ${primary:0:12}...${primary: -6}"
echo "Pool: $((${#keys[@]} - 1)) additional keys"

#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir> [repo-id] [commit-message]"}
REPO=${2:-"thewh1teagle/bert-char-he"}
MESSAGE=${3:-"add weights"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEOBERT_DIR="$SCRIPT_DIR/../src/neobert"

# Stage custom-code files at checkpoint root so Transformers can load them via `trust_remote_code=True`.
cp "$NEOBERT_DIR/model.py" "$CHECKPOINT/model.py"
cp "$NEOBERT_DIR/rotary.py" "$CHECKPOINT/rotary.py"

# Patch auto_map into config.json so AutoConfig/AutoModel can resolve the custom architecture.
uv run python - <<EOF
import json, pathlib
config_path = pathlib.Path("$CHECKPOINT/config.json")
config = json.loads(config_path.read_text())
config["auto_map"] = {
    "AutoConfig": "model.NeoBERTConfig",
    "AutoModel": "model.NeoBERT",
    "AutoModelForMaskedLM": "model.NeoBERTLMHead",
}
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
print("Patched auto_map into config.json")
EOF

# Upload everything in one commit
uv run hf upload "$REPO" "$CHECKPOINT" . --commit-message "$MESSAGE"

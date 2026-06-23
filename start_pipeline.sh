#!/bin/bash
# start_pipeline.sh — Start the enriched biotic interaction pipeline (port 8002)
# Existing ensemble API (port 8001) is NOT affected.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../MPvenv/bin/activate"

if [ -f "$VENV" ]; then
    source "$VENV"
fi

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

echo "Starting Biotic Interaction Pipeline API on port 8002..."
echo "  NER + GloBI term scan + Lexicon + ML + Outcome codes"
echo "  Original ensemble API (port 8001) is unaffected."
echo ""

cd "$SCRIPT_DIR"
uvicorn api.fastapi_pipeline:app \
    --host 0.0.0.0 \
    --port 8002 \
    --workers 1 \
    --log-level info

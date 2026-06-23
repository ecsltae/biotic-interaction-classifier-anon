#!/bin/bash
# Start the enriched pipeline (port 8002) with FLAN-T5 enriched as Layer 3.
# Falls back to discriminative ensemble if model not yet trained.
#
# Usage:
#   bash classifier/start_pipeline_generative.sh
#   bash classifier/start_pipeline_generative.sh /path/to/custom/flan_t5_model
#
# The FLAN-T5 enriched model is built by:
#   python classifier/src/models/flan_t5_enriched.py \
#     --train classifier/data/training/training_data_v12.csv --epochs 5

set -e
cd "$(dirname "$0")/.."

source /path/to/MetaP/MPvenv/bin/activate

GENERATIVE_MODEL_PATH="${1:-/path/to/MetaP/classifier/models/flan_t5_enriched}"
export GENERATIVE_MODEL_PATH

export PYTHONPATH="/path/to/MetaP/classifier/src:/path/to/MetaP/classifier"

if [ -f "$GENERATIVE_MODEL_PATH/config.json" ]; then
    echo "Generative model found: $GENERATIVE_MODEL_PATH"
    echo "Layer 3 will use FLAN-T5 enriched"
else
    echo "WARNING: Generative model not found at $GENERATIVE_MODEL_PATH"
    echo "Train it first with:"
    echo "  python classifier/src/models/flan_t5_enriched.py --train classifier/data/training/training_data_v12.csv"
    echo "Falling back to discriminative ensemble..."
fi

echo "Starting enriched pipeline (port 8002)..."
uvicorn classifier.api.fastapi_pipeline:app \
    --host 0.0.0.0 \
    --port 8002 \
    --workers 1 \
    --log-level info

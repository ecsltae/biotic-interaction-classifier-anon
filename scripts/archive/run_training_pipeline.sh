#!/bin/bash
# =============================================================================
# HIGH-PRECISION ENSEMBLE TRAINING PIPELINE
# =============================================================================
#
# Requirements:
#   - 80GB GPU (A100 or similar)
#   - Python environment with transformers, torch, sklearn
#
# Usage:
#   ./run_training_pipeline.sh
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="/path/to/MetaP/classifier/scripts"
VENV="/path/to/MetaP/MPvenv/bin/activate"

echo "============================================================"
echo "HIGH-PRECISION ENSEMBLE TRAINING PIPELINE"
echo "============================================================"
echo ""

# Activate environment
source $VENV

# Check GPU
echo "[1] Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"
echo ""

# Prepare dataset
echo "[2] Preparing precision-focused dataset..."
python3 $SCRIPT_DIR/prepare_precision_dataset.py
echo ""

# Train ensemble
echo "[3] Training precision ensemble..."
python3 $SCRIPT_DIR/train_precision_ensemble.py
echo ""

echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Models saved to: /path/to/MetaP/classifier/models/precision_ensemble/"
echo ""
echo "To start the API:"
echo "  python3 /path/to/MetaP/classifier/api/ensemble_api.py"
echo ""

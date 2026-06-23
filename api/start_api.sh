#!/bin/bash
# Start the FastAPI ensemble classifier in background
# The API will persist after closing the VM

cd /path/to/MetaP/classifier/api

# Kill any existing API process
pkill -f "fastapi_ensemble.py" 2>/dev/null
pkill -f "uvicorn.*8000" 2>/dev/null

# Wait a moment
sleep 2

# Create logs directory
mkdir -p /path/to/MetaP/classifier/logs

# Use the MetaP venv
PYTHON=/path/to/MetaP/MPvenv/bin/python

# Start with nohup
echo "Starting FastAPI Ensemble API..."
nohup $PYTHON /path/to/MetaP/classifier/api/fastapi_ensemble.py > /path/to/MetaP/classifier/logs/api.log 2>&1 &

# Get the PID
API_PID=$!
echo $API_PID > /path/to/MetaP/classifier/api/api.pid

# Wait for startup (models take time to load)
echo "Loading models... (please wait)"
sleep 15

# Get IP
IP=$(hostname -I | awk '{print $1}')

echo ""
echo "============================================"
echo "API STARTED"
echo "============================================"
echo "PID: $API_PID"
echo "URL: http://${IP}:8000"
echo ""
echo "Endpoints:"
echo "  GET  /          - API info"
echo "  GET  /health    - Health check"
echo "  POST /predict   - Single prediction"
echo "  POST /predict_batch - Batch prediction"
echo ""
echo "Example usage:"
echo '  curl -X POST "http://'${IP}':8000/predict_batch" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"sentences": ["bacteria infects host", "the sky is blue"]}'"'"
echo ""
echo "Log file: /path/to/MetaP/classifier/logs/api.log"
echo "============================================"

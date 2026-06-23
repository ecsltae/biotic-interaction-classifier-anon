#!/bin/bash
# Start the Biotic Interaction Classifier API
# Runs in background, survives logout, accessible to colleagues

cd /path/to/MetaP/classifier

# Kill any existing API on port 8001
pkill -f "uvicorn.*8001" 2>/dev/null || true
sleep 2

# Start API with nohup
echo "Starting Biotic Interaction Classifier API..."
echo "Model: SciBERT (F1=0.774, Precision=0.783)"
echo "Listening on: http://0.0.0.0:8001"

nohup python api_scibert.py > logs/api_scibert.log 2>&1 &

echo "API started with PID: $!"
echo ""
echo "To check status: curl http://localhost:8001/health"
echo "To view logs: tail -f logs/api_scibert.log"
echo "To stop: pkill -f 'uvicorn.*8001'"
echo ""
echo "Colleagues can access at: http://$(hostname -I | awk '{print $1}'):8001"

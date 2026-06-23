#!/usr/bin/env bash
# Start the Multi-task BiomedBERT API on port 8003
# Replaces distilled_BiomedBERT_v2 — new best model (EP F1=0.868)
# Survives logout (nohup), accessible to colleagues (0.0.0.0)

set -e
cd /path/to/MetaP
source MPvenv/bin/activate

PORT=8003
LOG="classifier/logs/api_distilled.log"
mkdir -p classifier/logs

# Kill any existing instance on this port
pkill -f "uvicorn.*$PORT" 2>/dev/null || true
sleep 1

echo "Starting Multi-task BiomedBERT API (full_typed_a05_ner2)..."
echo "  Model: multitask/full_typed_a05_ner2 (NER scheme=full_typed, α=0.5, 2ep NER pretrain)"
echo "  EP-relax F1=0.868 | AUC=0.887 | beats ensemble (F1=0.857)"
echo "  Port: $PORT"
echo "  Log:  $LOG"
echo ""

nohup python -u classifier/api/fastapi_multitask.py \
    >> "$LOG" 2>&1 &

PID=$!
echo "Started PID=$PID"
echo ""

# Wait for it to come up
for i in $(seq 1 15); do
    sleep 2
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✓ API is up."
        echo ""
        echo "  Local:      http://localhost:$PORT"
        echo "  Colleagues: http://$(hostname -I | awk '{print $1}'):$PORT"
        echo "  Docs:       http://$(hostname -I | awk '{print $1}'):$PORT/docs"
        echo ""
        echo "Example:"
        echo "  curl -s -X POST http://localhost:$PORT/predict \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"text\": \"Wolbachia infects Drosophila melanogaster.\"}' | python3 -m json.tool"
        echo ""
        echo "To stop:    pkill -f 'uvicorn.*$PORT'"
        echo "To monitor: tail -f $LOG"
        exit 0
    fi
done

echo "WARNING: API did not respond after 30s. Check $LOG"
tail -20 "$LOG"
exit 1

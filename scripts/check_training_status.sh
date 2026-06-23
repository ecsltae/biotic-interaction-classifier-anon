#!/bin/bash
# Check training status

LOG_FILE="/path/to/MetaP/classifier/logs/precision_training.log"

echo "============================================"
echo "PRECISION ENSEMBLE TRAINING STATUS"
echo "============================================"
echo ""

# Check if process is running
PID=$(pgrep -f "train_precision_ensemble.py" | head -1)
if [ -n "$PID" ]; then
    echo "✓ Training is RUNNING (PID: $PID)"
else
    echo "✗ Training is NOT RUNNING"
fi
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""

# Latest log entries
echo "Latest log entries:"
echo "-------------------"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE" | grep -E "(loss|epoch|TRAINING|precision|recall|f1|F1|Best|Saved|Complete)" | tail -15
else
    echo "Log file not found: $LOG_FILE"
fi
echo ""

# Log file size
if [ -f "$LOG_FILE" ]; then
    SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LINES=$(wc -l < "$LOG_FILE")
    echo "Log file: $SIZE ($LINES lines)"
fi
echo ""
echo "============================================"

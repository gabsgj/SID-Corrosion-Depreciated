#!/bin/bash
# Monitor training progress
# Usage: ./monitor_training.sh

LOG_FILE="$HOME/Projects/Corrosion/training_workspace/training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "No log file found at $LOG_FILE"
    echo "Training may not have started yet."
    exit 1
fi

echo "Monitoring training log..."
echo "Press Ctrl+C to exit"
echo ""
tail -f "$LOG_FILE"

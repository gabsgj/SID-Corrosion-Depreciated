#!/usr/bin/env python3
"""
Start YOLO11m training in background
"""

import subprocess
import sys
import os

# Change to project directory
os.chdir('/Users/gabriel/Projects/Corrosion')

# Run training with output to file
cmd = [
    '/Users/gabriel/Projects/Corrosion/venv_yolo/bin/python',
    '/Users/gabriel/Projects/Corrosion/train_yolo11m.py'
]

print("Starting YOLO11m training in background...")
print("Training will continue even if you close this terminal.")
print("\nTo check progress:")
print("  tail -f /Users/gabriel/Projects/Corrosion/training.log")
print("\nTo stop training:")
print("  pkill -f train_yolo11m.py")

# Run in background with output to log file
with open('/Users/gabriel/Projects/Corrosion/training.log', 'w') as log:
    process = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True  # This makes it continue after terminal closes
    )

print(f"\n✓ Training started with PID: {process.pid}")
print(f"✓ Log file: /Users/gabriel/Projects/Corrosion/training.log")

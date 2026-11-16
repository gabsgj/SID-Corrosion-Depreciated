#!/usr/bin/env python3
"""
Training script with file logging for monitoring with:
    tail -f ~/Projects/Corrosion/training_workspace/training.log
"""

import logging
import sys
from pathlib import Path

# Setup logging
workspace = Path.home() / 'Projects' / 'Corrosion' / 'training_workspace'
workspace.mkdir(parents=True, exist_ok=True)
log_file = workspace / 'training.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# Now run the actual training
exec(open('train_corrosion_yolo11.py').read())

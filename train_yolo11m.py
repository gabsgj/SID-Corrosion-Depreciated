#!/usr/bin/env python3
"""
YOLO11m Corrosion Detection Training Script
Optimized for MacBook Air M5 with MPS
"""

import os
import sys
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from ultralytics import YOLO
import numpy as np

print("="*60)
print("YOLO11m Corrosion Detection Training - MacBook M5")
print("="*60)

# Check MPS availability
print(f"\nDevice Information:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  MPS available: {torch.backends.mps.is_available()}")
print(f"  MPS built: {torch.backends.mps.is_built()}")

# Configure device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"  Using device: {device}")

# Paths
WORKSPACE = Path('/Users/gabriel/Projects/Corrosion')
DATASET_PATH = WORKSPACE / 'corrosion_dataset'
RESULTS_PATH = WORKSPACE / 'training_results'
RESULTS_PATH.mkdir(exist_ok=True)

# Training state file
STATE_FILE = RESULTS_PATH / 'training_state.json'

class TrainingState:
    """Manages training state for resume capability"""
    
    def __init__(self, state_file):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'model_trained': False,
            'epochs_completed': 0,
            'best_mAP50': 0,
            'timestamp': None
        }
    
    def save_state(self):
        self.state['timestamp'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_model_trained(self, epochs_completed, best_mAP50):
        self.state['model_trained'] = True
        self.state['epochs_completed'] = epochs_completed
        self.state['best_mAP50'] = best_mAP50
        self.save_state()
    
    def print_status(self):
        print(f"\nTraining State:")
        print(f"  Model trained: {self.state['model_trained']}")
        print(f"  Epochs completed: {self.state['epochs_completed']}")
        print(f"  Best mAP50: {self.state['best_mAP50']:.4f}")
        if self.state['timestamp']:
            print(f"  Last update: {self.state['timestamp']}")
        print()

# Initialize state manager
state = TrainingState(STATE_FILE)
state.print_status()

# Training configuration - OPTIMIZED FOR MACBOOK M5
TRAINING_CONFIG = {
    # Data
    'data': str(DATASET_PATH / 'data.yaml') if (DATASET_PATH / 'data.yaml').exists() else None,
    
    # Training params - optimized for speed
    'epochs': 50,           # Reduced from 100 for faster training
    'imgsz': 320,            # Reduced from 640 (4x faster training)
    'batch': 16,             # Good balance for 32GB RAM
    'device': device,        # MPS for Mac
    
    # Optimizer
    'optimizer': 'auto',     # Auto-select (AdamW for segmentation)
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    
    # Augmentation (reduce for speed)
    'degrees': 10.0,         # Reduced from 15
    'translate': 0.1,        # Reduced from 0.2
    'scale': 0.5,
    'shear': 2.0,            # Reduced from 5
    'flipud': 0.5,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'mosaic': 0.5,           # Reduced from 1.0
    'mixup': 0.0,            # Disabled for speed
    'copy_paste': 0.0,       # Disabled for speed
    'close_mosaic': 5,       # Close mosaic early
    
    # Other settings
    'patience': 15,          # Early stopping
    'save_period': 5,        # Save every 5 epochs
    'cache': True,           # Cache images in RAM (32GB available)
    'cos_lr': True,          # Cosine LR schedule
    'exist_ok': True,
    'pretrained': True,
    'verbose': True,
    'seed': 42,
    'workers': 8,            # More workers with 32GB RAM
    'plots': True,
    'val': True,
}

print("Training Configuration:")
for key, value in TRAINING_CONFIG.items():
    if value is not None:
        print(f"  {key}: {value}")
print()

def train_yolo11m():
    """Train YOLO11m model"""
    
    print("="*60)
    print("Training YOLO11m Model")
    print("="*60)
    
    # Model configuration
    model_config = {
        'weights': 'yolo11m.pt',
        'name': 'yolo11m_corrosion',
        'epochs': TRAINING_CONFIG['epochs'],
        'batch': TRAINING_CONFIG['batch'],
    }
    
    # Check if already trained
    if state.state['model_trained']:
        print(f"Model already trained: {state.state['epochs_completed']} epochs completed")
        print(f"Best mAP50: {state.state['best_mAP50']:.4f}")
        return
    
    # Setup paths
    project_dir = RESULTS_PATH / 'runs'
    name = model_config['name']
    
    # Check for existing checkpoint
    checkpoint_dir = project_dir / name
    last_pt = checkpoint_dir / 'weights' / 'last.pt'
    
    resume_from = None
    if last_pt.exists():
        print(f"Found existing checkpoint: {last_pt}")
        resume_from = str(last_pt)
    
    # Load model
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from: {resume_from}")
        model = YOLO(resume_from)
        resume_flag = True
    else:
        print(f"Loading pretrained weights: {model_config['weights']}")
        model = YOLO(model_config['weights'])
        resume_flag = False
    
    # Prepare training arguments
    train_args = TRAINING_CONFIG.copy()
    train_args['project'] = str(project_dir)
    train_args['name'] = name
    train_args['resume'] = resume_flag
    
    # Check if data.yaml exists
    if train_args['data'] is None or not Path(train_args['data']).exists():
        print(f"ERROR: data.yaml not found at {train_args['data']}")
        print("Please ensure your dataset is in the correct location:")
        print(f"  Expected: {DATASET_PATH}/data.yaml")
        print(f"\nDataset should have structure:")
        print(f"  {DATASET_PATH}/")
        print(f"    data.yaml")
        print(f"    train/images/")
        print(f"    train/labels/")
        print(f"    valid/images/")
        print(f"    valid/labels/")
        print(f"    test/images/ (optional)")
        print(f"    test/labels/ (optional)")
        return None
    
    # Train
    try:
        print(f"\nStarting training for {model_config['epochs']} epochs...")
        print(f"Device: {device}")
        print(f"Image size: {TRAINING_CONFIG['imgsz']}")
        print(f"Batch size: {TRAINING_CONFIG['batch']}")
        print()
        
        results = model.train(**train_args)
        
        # Get best mAP
        best_mAP50 = results.results_dict.get('metrics/mAP50(M)', 0)
        epochs_completed = results.epochs
        
        # Mark as trained
        state.mark_model_trained(epochs_completed, best_mAP50)
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"{'='*60}")
        print(f"  Epochs completed: {epochs_completed}")
        print(f"  Best mAP50: {best_mAP50:.4f}")
        print(f"  Results saved to: {checkpoint_dir}")
        
        return results
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        print(f"You can resume by running this script again!")
        raise

def evaluate_model():
    """Evaluate trained model"""
    
    model_path = RESULTS_PATH / 'runs' / 'yolo11m_corrosion' / 'weights' / 'best.pt'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    print(f"\n{'='*60}")
    print("Evaluating Model")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    
    # Run validation
    results = model.val(
        data=TRAINING_CONFIG['data'],
        imgsz=TRAINING_CONFIG['imgsz'],
        batch=TRAINING_CONFIG['batch'],
        device=device,
        split='test' if (DATASET_PATH / 'test' / 'images').exists() else 'val',
        save_json=True,
    )
    
    # Extract metrics
    metrics = {
        'mAP50': results.results_dict.get('metrics/mAP50(M)', 0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(M)', 0),
        'Precision': results.results_dict.get('metrics/precision(M)', 0),
        'Recall': results.results_dict.get('metrics/recall(M)', 0),
    }
    
    print(f"\nEvaluation Results:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    
    return metrics

def export_model():
    """Export model to different formats"""
    
    model_path = RESULTS_PATH / 'runs' / 'yolo11m_corrosion' / 'weights' / 'best.pt'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    print(f"\n{'='*60}")
    print("Exporting Model")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    
    # Export to ONNX
    print("Exporting to ONNX...")
    try:
        onnx_path = model.export(format='onnx', imgsz=TRAINING_CONFIG['imgsz'], dynamic=True)
        print(f"  Exported to ONNX: {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
    
    # Export to TorchScript
    print("Exporting to TorchScript...")
    try:
        torchscript_path = model.export(format='torchscript', imgsz=TRAINING_CONFIG['imgsz'])
        print(f"  Exported to TorchScript: {torchscript_path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
    
    print("Export completed!")

def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("YOLO11m Corrosion Detection Training")
    print("MacBook Air M5 - Optimized Configuration")
    print("="*60 + "\n")
    
    # Check dataset
    if not DATASET_PATH.exists():
        print(f"Creating dataset directory: {DATASET_PATH}")
        DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    # Check if data.yaml exists
    if not (DATASET_PATH / 'data.yaml').exists():
        print("="*60)
        print("DATASET NOT FOUND!")
        print("="*60)
        print(f"\nPlease prepare your dataset at:")
        print(f"  {DATASET_PATH}")
        print(f"\nExpected structure:")
        print(f"  {DATASET_PATH}/")
        print(f"    data.yaml          # Dataset configuration")
        print(f"    train/")
        print(f"      images/          # Training images")
        print(f"      labels/          # Training labels (YOLO format)")
        print(f"    valid/")
        print(f"      images/          # Validation images")
        print(f"      labels/          # Validation labels")
        print(f"\ndata.yaml example:")
        print("  path: /Users/gabriel/Projects/Corrosion/corrosion_dataset")
        print("  train: train/images")
        print("  val: valid/images")
        print("  nc: 9")
        print("  names: ['corrosion', 'mild_corrosion', 'moderate_corrosion', 'severe_corrosion', 'pitting_corrosion', 'uniform_corrosion', 'rebar_corrosion', 'steel_corrosion', 'crack']")
        return
    
    # Train
    results = train_yolo11m()
    
    if results:
        # Evaluate
        evaluate_model()
        
        # Export
        export_model()
        
        print(f"\n{'='*60}")
        print("ALL TASKS COMPLETED!")
        print(f"{'='*60}")
        print(f"Results saved to: {RESULTS_PATH}")

if __name__ == '__main__':
    main()

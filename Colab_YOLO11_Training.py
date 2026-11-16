# Corrosion YOLO11 Training - Google Colab Notebook
# Optimized for Colab GPU (Tesla T4 or better)

# %% Setup Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
WORKSPACE = '/content/corrosion_workspace'
os.makedirs(WORKSPACE, exist_ok=True)
os.chdir(WORKSPACE)

# Google Drive paths for persistence
DRIVE_BASE = '/content/drive/MyDrive/YOLO_Corrosion_Training'
DRIVE_DATASETS = f'{DRIVE_BASE}/datasets'
DRIVE_MODELS = f'{DRIVE_BASE}/models'
DRIVE_RESULTS = f'{DRIVE_BASE}/results'

for path in [DRIVE_BASE, DRIVE_DATASETS, DRIVE_MODELS, DRIVE_RESULTS]:
    os.makedirs(path, exist_ok=True)

print(f"Working in: {WORKSPACE}")
print(f"Drive base: {DRIVE_BASE}")

# %% Setup Cell 2: Install Dependencies
!pip install -q ultralytics roboflow

import yaml
import json
import shutil
from pathlib import Path
from glob import glob
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% Setup Cell 3: Dataset Download Configuration
from roboflow import Roboflow

# Roboflow API Key - REPLACE WITH YOURS
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"  # Get from https://app.roboflow.com/api-key

# Datasets to download
DATASETS_TO_DOWNLOAD = [
    ('cawilcorrosionsegmentation/instance-corrosion', 2, 'instance_corrosion'),
    ('yatishs-workspace/corrosion-detection-z8iw3', 1, 'corrosion_detection_large'),
    ('thyssenkrupp/corrosion-whmsc', 1, 'thyssenkrupp_steel'),
    ('university-of-tebessa/corrosion-eh3ms', 1, 'university_steel_corrosion'),
    ('corrosion-l4cqs/corrosion-detection-npdhd', 4, 'large_corrosion_detection'),
    ('identification-pittin-corrosion/corrosion-segmentation-rsfbe', 1, 'multi_type_corrosion'),
    ('godson-0hjq0/corrosion-detection-0pt6o', 1, 'corrosion_types_4class'),
    ('scaledge-ztihl/corrosion-detection-obirw', 1, 'scaledge_corrosion'),
    ('drima/corrosion-zjnrh', 1, 'drima_corrosion'),
]

# %% Setup Cell 4: Download Datasets Function
def download_from_roboflow(api_key, workspace_project, version, dataset_name, output_dir):
    """Download a dataset from Roboflow"""
    try:
        print(f"\nDownloading {dataset_name}...")
        rf = Roboflow(api_key=api_key)
        workspace, project = workspace_project.split('/')
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8", location=output_dir)
        print(f"✓ Downloaded {dataset_name}")
        return True
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return False

def download_all_datasets():
    """Download all corrosion datasets"""
    datasets_dir = f'{WORKSPACE}/datasets'
    os.makedirs(datasets_dir, exist_ok=True)
    
    downloaded = []
    
    for workspace_project, version, dataset_name in DATASETS_TO_DOWNLOAD:
        output_dir = f'{datasets_dir}/{dataset_name}'
        
        # Check if already exists
        if os.path.exists(output_dir):
            print(f"{dataset_name}: Already exists, skipping...")
            downloaded.append(dataset_name)
            continue
        
        success = download_from_roboflow(
            ROBOFLOW_API_KEY, workspace_project, version, dataset_name, output_dir
        )
        
        if success:
            downloaded.append(dataset_name)
            # Copy to Drive
            drive_path = f'{DRIVE_DATASETS}/{dataset_name}'
            if not os.path.exists(drive_path):
                shutil.copytree(output_dir, drive_path)
                print(f"  Copied to Drive")
    
    return downloaded

# %% Cell 5: Download Datasets (Run if needed)
# Set your API key first above, then run this cell
if ROBOFLOW_API_KEY != "YOUR_API_KEY_HERE":
    downloaded = download_all_datasets()
    print(f"\n✓ Downloaded {len(downloaded)} datasets")
else:
    print("Please set ROBOFLOW_API_KEY first!")
    print("Get it from: https://app.roboflow.com/api-key")

# %% Cell 6: Analyze Datasets
def analyze_dataset(dataset_path):
    """Analyze a YOLO format dataset"""
    info = {
        'name': os.path.basename(dataset_path),
        'train_images': 0, 'val_images': 0, 'test_images': 0,
        'classes': [], 'total_annotations': 0
    }
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            info['classes'] = data.get('names', [])
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(dataset_path, split, 'images')
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if split == 'train': info['train_images'] = count
            elif split == 'valid': info['val_images'] = count
            else: info['test_images'] = count
    
    return info

def analyze_all_datasets():
    """Analyze all downloaded datasets"""
    datasets_dir = f'{WORKSPACE}/datasets'
    datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    
    print("\n=== Dataset Analysis ===")
    total_train = total_val = total_test = 0
    
    for dataset_name in datasets:
        path = os.path.join(datasets_dir, dataset_name)
        info = analyze_dataset(path)
        total_train += info['train_images']
        total_val += info['val_images']
        total_test += info['test_images']
        
        print(f"\n{dataset_name}:")
        print(f"  Classes: {info['classes'][:3]}...")  # Show first 3
        print(f"  Train: {info['train_images']}, Val: {info['val_images']}, Test: {info['test_images']}")
    
    print(f"\n=== Total Statistics ===")
    print(f"Train images: {total_train}")
    print(f"Val images: {total_val}")
    print(f"Test images: {total_test}")
    print(f"Total: {total_train + total_val + total_test}")
    
    return datasets

# Run analysis
datasets = analyze_all_datasets()

# %% Cell 7: Merge Datasets
UNIFIED_CLASSES = ['corrosion', 'mild_corrosion', 'moderate_corrosion', 
                   'severe_corrosion', 'pitting_corrosion', 'uniform_corrosion',
                   'rebar_corrosion', 'steel_corrosion', 'crack']

def standardize_dataset(dataset_path, unified_classes, output_path):
    """Convert dataset to unified class names"""
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        return
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    original_classes = data.get('names', [])
    
    # Create class mapping
    class_mapping = {}
    for i, orig_class in enumerate(original_classes):
        orig_lower = orig_class.lower()
        
        if any(x in orig_lower for x in ['corrosion', 'rust', 'corroded']):
            if any(x in orig_lower for x in ['severe', 'poor', 'high']) or '4' in orig_lower:
                class_mapping[i] = 'severe_corrosion'
            elif any(x in orig_lower for x in ['moderate', 'fair', 'medium']) or '3' in orig_lower:
                class_mapping[i] = 'moderate_corrosion'
            elif any(x in orig_lower for x in ['mild', 'low', 'light']) or '2' in orig_lower:
                class_mapping[i] = 'mild_corrosion'
            elif 'pitting' in orig_lower:
                class_mapping[i] = 'pitting_corrosion'
            elif 'uniform' in orig_lower:
                class_mapping[i] = 'uniform_corrosion'
            elif 'crevice' in orig_lower:
                class_mapping[i] = 'crevice_corrosion'
            else:
                class_mapping[i] = 'corrosion'
        elif 'crack' in orig_lower:
            class_mapping[i] = 'crack'
        else:
            class_mapping[i] = 'corrosion'
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        src_img_dir = os.path.join(dataset_path, split, 'images')
        src_lbl_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(src_img_dir):
            continue
        
        dst_img_dir = os.path.join(output_path, split, 'images')
        dst_lbl_dir = os.path.join(output_path, split, 'labels')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        for img_file in tqdm(os.listdir(src_img_dir), desc=f"Processing {split}", leave=False):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            src_img = os.path.join(src_img_dir, img_file)
            new_img_name = f"{os.path.basename(dataset_path)}_{img_file}"
            dst_img = os.path.join(dst_img_dir, new_img_name)
            shutil.copy2(src_img, dst_img)
            
            lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_lbl = os.path.join(src_lbl_dir, lbl_file)
            dst_lbl = os.path.join(dst_lbl_dir, new_img_name.rsplit('.', 1)[0] + '.txt')
            
            if os.path.exists(src_lbl):
                with open(src_lbl, 'r') as f_in, open(dst_lbl, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        orig_class = int(parts[0])
                        if orig_class in class_mapping:
                            unified_class = class_mapping[orig_class]
                            if unified_class in unified_classes:
                                new_class_id = unified_classes.index(unified_class)
                                f_out.write(f"{new_class_id} {' '.join(parts[1:])}\n")

def merge_datasets():
    """Merge all datasets into unified format"""
    datasets_dir = f'{WORKSPACE}/datasets'
    merged_path = f'{WORKSPACE}/merged_corrosion_dataset'
    
    print(f"Creating unified dataset with {len(UNIFIED_CLASSES)} classes...")
    print(f"Classes: {UNIFIED_CLASSES}")
    
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
    os.makedirs(merged_path, exist_ok=True)
    
    datasets = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir) 
                if os.path.isdir(os.path.join(datasets_dir, d))]
    
    for dataset_path in tqdm(datasets, desc="Merging datasets"):
        standardize_dataset(dataset_path, UNIFIED_CLASSES, merged_path)
    
    # Create data.yaml
    data_yaml = {
        'path': merged_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(UNIFIED_CLASSES),
        'names': UNIFIED_CLASSES
    }
    
    yaml_path = os.path.join(merged_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Count statistics
    total_train = len([f for f in os.listdir(os.path.join(merged_path, 'train', 'images')) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_val = len([f for f in os.listdir(os.path.join(merged_path, 'valid', 'images')) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_test = len([f for f in os.listdir(os.path.join(merged_path, 'test', 'images')) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"\n✓ Merged dataset created:")
    print(f"  Train: {total_train}")
    print(f"  Val: {total_val}")
    print(f"  Test: {total_test}")
    print(f"  Total: {total_train + total_val + total_test}")
    
    return merged_path

MERGED_DATASET_PATH = merge_datasets()

# %% Cell 8: Training Configuration
from ultralytics import YOLO

TRAINING_CONFIG = {
    'data': os.path.join(MERGED_DATASET_PATH, 'data.yaml'),
    'epochs': 100,  # Full training
    'imgsz': 640,  # Higher resolution
    'batch': 16,
    'device': 0,  # GPU
    'patience': 20,
    'save_period': 10,
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'auto',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'cache': True,  # Cache in RAM
    'cos_lr': True,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'degrees': 15.0,
    'translate': 0.2,
    'scale': 0.5,
    'shear': 5.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'mosaic': 1.0,
    'mixup': 0.2,
    'close_mosaic': 10,
    'plots': True,
    'val': True,
    'verbose': True,
}

print("Training configuration:")
for k, v in TRAINING_CONFIG.items():
    print(f"  {k}: {v}")

# %% Cell 9: Train YOLO11m
print("\n" + "="*60)
print("Starting YOLO11m Training")
print("="*60)

model = YOLO('yolo11m.pt')

# Prepare args
train_args = TRAINING_CONFIG.copy()
train_args['project'] = f'{WORKSPACE}/runs'
train_args['name'] = 'yolo11m_corrosion_colab'

# Train
results = model.train(**train_args)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# %% Cell 10: Evaluate Model
print("\nEvaluating model...")
metrics = model.val(
    data=os.path.join(MERGED_DATASET_PATH, 'data.yaml'),
    imgsz=640,
    batch=16,
    device=0,
    split='test'
)

print(f"\nFinal Metrics:")
print(f"  mAP50: {metrics.results_dict.get('metrics/mAP50(M)', 0):.4f}")
print(f"  mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(M)', 0):.4f}")
print(f"  Precision: {metrics.results_dict.get('metrics/precision(M)', 0):.4f}")
print(f"  Recall: {metrics.results_dict.get('metrics/recall(M)', 0):.4f}")

# %% Cell 11: Export Model
print("\nExporting model...")

# Export to ONNX
onnx_path = model.export(format='onnx', imgsz=640, dynamic=True)
print(f"ONNX: {onnx_path}")

# Export to TorchScript
torchscript_path = model.export(format='torchscript', imgsz=640)
print(f"TorchScript: {torchscript_path}")

# %% Cell 12: Save to Google Drive
print("\nSaving results to Google Drive...")

# Copy best model
best_model_path = f'{WORKSPACE}/runs/yolo11m_corrosion_colab/weights/best.pt'
shutil.copy2(best_model_path, f'{DRIVE_MODELS}/yolo11m_corrosion_best.pt')

# Copy results
results_path = f'{WORKSPACE}/runs/yolo11m_corrosion_colab'
for file in ['results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png']:
    src = os.path.join(results_path, file)
    if os.path.exists(src):
        shutil.copy2(src, DRIVE_RESULTS)

print(f"\n✓ Results saved to Google Drive:")
print(f"  Model: {DRIVE_MODELS}/yolo11m_corrosion_best.pt")
print(f"  Results: {DRIVE_RESULTS}/")

print("\n" + "="*60)
print("ALL TASKS COMPLETED!")
print("="*60)

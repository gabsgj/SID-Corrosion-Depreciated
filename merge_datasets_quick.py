#!/usr/bin/env python3
"""
Merge downloaded corrosion datasets into a unified dataset
Uses only datasets that have been downloaded so far
"""

import os
import yaml
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def analyze_dataset(dataset_path):
    """Analyze a YOLO format dataset"""
    info = {
        'path': dataset_path,
        'name': os.path.basename(dataset_path),
        'train_images': 0,
        'val_images': 0,
        'test_images': 0,
        'classes': [],
        'total_annotations': 0,
        'task': 'unknown'
    }
    
    # Read data.yaml if exists
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            info['classes'] = data.get('names', [])
            info['nc'] = data.get('nc', len(info['classes']))
    
    # Count images
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(dataset_path, split, 'images')
        if os.path.exists(img_dir):
            count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if split == 'train':
                info['train_images'] = count
            elif split == 'valid':
                info['val_images'] = count
            else:
                info['test_images'] = count
        
        # Count labels
        label_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(label_dir):
            for lbl_file in os.listdir(label_dir):
                if lbl_file.endswith('.txt'):
                    lbl_path = os.path.join(label_dir, lbl_file)
                    with open(lbl_path, 'r') as f:
                        info['total_annotations'] += len(f.readlines())
    
    return info

def standardize_dataset(dataset_path, unified_classes, output_path):
    """Convert dataset to unified class names"""
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Warning: No data.yaml found in {dataset_path}")
        return
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    original_classes = data.get('names', [])
    
    # Simple mapping - all corrosion-related classes map to 'corrosion'
    class_mapping = {}
    for i, orig_class in enumerate(original_classes):
        orig_lower = orig_class.lower()
        
        if any(x in orig_lower for x in ['corrosion', 'rust', 'corroded']):
            class_mapping[i] = 'corrosion'
        elif any(x in orig_lower for x in ['rebar', 'bar', 'steel']):
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
        
        img_files = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in tqdm(img_files, desc=f"Processing {split}", leave=False):
            # Copy image with unique name
            src_img = os.path.join(src_img_dir, img_file)
            dataset_name = os.path.basename(dataset_path)
            new_img_name = f"{dataset_name}_{img_file}"
            dst_img = os.path.join(dst_img_dir, new_img_name)
            shutil.copy2(src_img, dst_img)
            
            # Process labels
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

def main():
    print("="*60)
    print("Merging Downloaded Corrosion Datasets")
    print("="*60)
    
    # Paths
    datasets_dir = Path('/Users/gabriel/Projects/Corrosion/datasets')
    merged_dataset_path = Path('/Users/gabriel/Projects/Corrosion/corrosion_dataset')
    
    # Simple unified classes
    UNIFIED_CLASSES = ['corrosion', 'crack']
    
    print(f"\nUnified classes: {UNIFIED_CLASSES}")
    
    # Find all downloaded datasets
    if not datasets_dir.exists():
        print(f"✗ Datasets directory not found: {datasets_dir}")
        return False
    
    datasets = [d for d in datasets_dir.iterdir() if d.is_dir() and (d / 'data.yaml').exists()]
    
    if not datasets:
        print("\n✗ No datasets found! Please wait for downloads to complete.")
        return False
    
    print(f"\nFound {len(datasets)} downloaded datasets:")
    for dataset in datasets:
        print(f"  - {dataset.name}")
    
    # Analyze datasets
    print("\nAnalyzing datasets...")
    total_train = total_val = total_test = 0
    
    for dataset in datasets:
        info = analyze_dataset(str(dataset))
        total_train += info['train_images']
        total_val += info['val_images']
        total_test += info['test_images']
        print(f"\n{dataset.name}:")
        print(f"  Classes: {info['classes'][:5]}...")  # Show first 5
        print(f"  Train: {info['train_images']}, Val: {info['val_images']}, Test: {info['test_images']}")
    
    # Create merged dataset
    print(f"\n{'='*60}")
    print("Creating unified dataset...")
    print(f"{'='*60}")
    
    if merged_dataset_path.exists():
        print(f"Removing existing merged dataset...")
        shutil.rmtree(merged_dataset_path)
    
    merged_dataset_path.mkdir(exist_ok=True)
    
    # Process each dataset
    for dataset in tqdm(datasets, desc="Merging datasets"):
        standardize_dataset(str(dataset), UNIFIED_CLASSES, str(merged_dataset_path))
    
    # Count merged dataset
    final_train = len([f for f in (merged_dataset_path / 'train' / 'images').iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']]) if (merged_dataset_path / 'train' / 'images').exists() else 0
    final_val = len([f for f in (merged_dataset_path / 'valid' / 'images').iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']]) if (merged_dataset_path / 'valid' / 'images').exists() else 0
    final_test = len([f for f in (merged_dataset_path / 'test' / 'images').iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']]) if (merged_dataset_path / 'test' / 'images').exists() else 0
    
    print(f"\n{'='*60}")
    print("Merged Dataset Statistics")
    print(f"{'='*60}")
    print(f"Train images: {final_train}")
    print(f"Val images: {final_val}")
    print(f"Test images: {final_test}")
    print(f"Total images: {final_train + final_val + final_test}")
    
    # Create data.yaml
    data_yaml = {
        'path': str(merged_dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(UNIFIED_CLASSES),
        'names': UNIFIED_CLASSES
    }
    
    yaml_path = merged_dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✓ Created data.yaml at {yaml_path}")
    print(f"\nDataset is ready for training!")
    print(f"Run: python train_yolo11m.py")
    
    return True

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Merge multiple corrosion datasets into a unified dataset
Optimized for YOLO training
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
    
    # Detect task type from label files
    sample_label = None
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(label_dir):
            label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            if label_files:
                sample_label = os.path.join(label_dir, label_files[0])
                break
    
    if sample_label:
        with open(sample_label, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) == 5:
                    info['task'] = 'detection'
                elif len(parts) > 5 and len(parts) % 2 == 1:
                    info['task'] = 'segmentation'
    
    return info

def standardize_dataset(dataset_path, unified_classes, output_path):
    """Convert dataset to unified class names and save to merged dataset"""
    
    # Read original data.yaml
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Warning: No data.yaml found in {dataset_path}")
        return
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    original_classes = data.get('names', [])
    
    # Create class mapping
    class_mapping = {}
    for i, orig_class in enumerate(original_classes):
        orig_lower = orig_class.lower()
        
        # Map to unified classes
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
        elif any(x in orig_lower for x in ['clean', 'no-corroded', 'non-corroded']):
            class_mapping[i] = 'no_corrosion'
        elif any(x in orig_lower for x in ['rebar', 'bar']):
            class_mapping[i] = 'rebar_corrosion'
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
            
            # Copy image
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
    print("Merging Corrosion Datasets")
    print("="*60)
    
    # Paths
    datasets_dir = Path('/Users/gabriel/Projects/Corrosion/datasets')
    merged_dataset_path = Path('/Users/gabriel/Projects/Corrosion/corrosion_dataset')
    
    # Define unified class names
    UNIFIED_CLASSES = [
        'corrosion',
        'mild_corrosion',
        'moderate_corrosion',
        'severe_corrosion',
        'pitting_corrosion',
        'uniform_corrosion',
        'rebar_corrosion',
        'steel_corrosion',
        'crack',
    ]
    
    print(f"\nUnified classes: {UNIFIED_CLASSES}")
    print(f"\nLooking for datasets in: {datasets_dir}")
    
    # Find all datasets
    datasets = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    if not datasets:
        print("\n✗ No datasets found! Please run download_datasets.py first.")
        return False
    
    print(f"\nFound {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"  - {dataset.name}")
    
    # Analyze datasets
    print("\nAnalyzing datasets...")
    dataset_infos = []
    for dataset in datasets:
        info = analyze_dataset(str(dataset))
        dataset_infos.append(info)
        print(f"\n{dataset.name}:")
        print(f"  Task: {info['task']}")
        print(f"  Classes: {info['classes']}")
        print(f"  Train: {info['train_images']}, Val: {info['val_images']}, Test: {info['test_images']}")
    
    # Create merged dataset
    print(f"\n{'='*60}")
    print("Creating unified dataset...")
    print(f"{'='*60}")
    
    merged_dataset_path.mkdir(exist_ok=True)
    
    # Process each dataset
    for dataset in tqdm(datasets, desc="Merging datasets"):
        standardize_dataset(str(dataset), UNIFIED_CLASSES, str(merged_dataset_path))
    
    # Count merged dataset
    total_train = len([f for f in (merged_dataset_path / 'train' / 'images').iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']]) if (merged_dataset_path / 'train' / 'images').exists() else 0
    total_val = len([f for f in (merged_dataset_path / 'valid' / 'images').iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']]) if (merged_dataset_path / 'valid' / 'images').exists() else 0
    total_test = len([f for f in (merged_dataset_path / 'test' / 'images').iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']]) if (merged_dataset_path / 'test' / 'images').exists() else 0
    
    print(f"\n{'='*60}")
    print("Merged Dataset Statistics")
    print(f"{'='*60}")
    print(f"Train images: {total_train}")
    print(f"Val images: {total_val}")
    print(f"Test images: {total_test}")
    print(f"Total images: {total_train + total_val + total_test}")
    
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
    
    return True

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Corrosion Detection Training for Bridges - YOLO11
Optimized for MacBook M5 with MPS
Fully resumable training script

Usage:
    source venv_yolo/bin/activate
    python train_corrosion_yolo11.py

To resume training after interruption:
    python train_corrosion_yolo11.py

To force restart:
    FORCE_RESTART=1 python train_corrosion_yolo11.py
"""

import os
import sys
import yaml
import json
import shutil
import random
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
CONFIG = {
    'ROBOFLOW_API_KEY': "5Zmgm6NCmNWapNJMTOtk",  # Your API key
    'WORKSPACE_ROOT': Path.home() / 'Projects' / 'Corrosion' / 'training_workspace',
    
    # Training settings (M5 optimized)
    'EPOCHS': 100,
    'PATIENCE': 30,
    'BATCH_SIZE': 8,  # Conservative for 32GB unified memory
    'IMAGE_SIZE': 640,
    'USE_MPS': True,
    
    # Class mapping
    'UNIFIED_CLASSES': [
        'rebar_corrosion',   # 0: rust staining + spalling from corroded rebar
        'steel_corrosion',   # 1: surface rust on steel beams, girders, bolts
        'rust_stain',        # 2: brown/orange streaking on concrete (early stage)
        'exposed_rebar'      # 3: rebar visible through spalled concrete
    ],
    
    # Dataset class mappings
    'CLASS_MAPPING': {
        'codebrim': {
            'corrosion': 'rust_stain',
            'exposed_bars': 'exposed_rebar',
            'spalling': 'rebar_corrosion',
        }
    },
    
    # Roboflow datasets
    'ROBOFLOW_DATASETS': [
        ('tcc-fi6cb/detection-corrosion', 1, 'detection_corrosion'),
        ('sutan-muhamad-sadam-awal-oo6pb/corrosion-detection-ofnck', 1, 'corrosion_detection_ofnck'),
    ],
    
    # Resume settings
    'FORCE_RESTART': os.environ.get('FORCE_RESTART', '0') == '1',
    'RUN_NAME': 'yolo11m_bridge_corrosion',
}

class CorrosionTrainer:
    """Main training class with full resume capability"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_paths()
        self.setup_device()
        self.import_ultralytics()
        
    def setup_paths(self):
        """Initialize directory structure"""
        self.workspace = Path(self.config['WORKSPACE_ROOT']).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.datasets_dir = self.workspace / 'datasets'
        self.models_dir = self.workspace / 'models'
        self.runs_dir = self.workspace / 'runs'
        self.merged_dataset_path = self.workspace / 'merged_dataset'
        
        for path in [self.datasets_dir, self.models_dir, self.runs_dir]:
            path.mkdir(exist_ok=True)
            
        print(f"✓ Workspace: {self.workspace}")
        print(f"  Datasets: {self.datasets_dir}")
        print(f"  Models: {self.models_dir}")
        print(f"  Runs: {self.runs_dir}")
        
    def setup_device(self):
        """Configure device (MPS for Mac, CPU fallback)"""
        import torch
        
        if self.config['USE_MPS'] and torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"✓ Using MPS (Metal Performance Shaders)")
        else:
            self.device = 'cpu'
            print(f"⚠ Using CPU (MPS not available)")
            
        print(f"  PyTorch: {torch.__version__}")
        
    def import_ultralytics(self):
        """Import ultralytics after device setup"""
        global YOLO
        from ultralytics import YOLO as YOLO_Class
        YOLO = YOLO_Class
        
    def download_codebrim(self) -> Path:
        """Download and extract CODEBRIM dataset from Zenodo"""
        codebrim_dir = self.datasets_dir / 'codebrim_original'
        zip_path = self.datasets_dir / 'CODEBRIM_original_images.zip'
        
        # Check if already extracted
        if codebrim_dir.exists() and any(codebrim_dir.iterdir()):
            print(f"✓ CODEBRIM already downloaded")
            return codebrim_dir
        
        # Download if needed
        if not zip_path.exists():
            print("Downloading CODEBRIM from Zenodo (8.3 GB)...")
            print("This may take 20-40 minutes depending on connection...")
            url = "https://zenodo.org/api/records/2620293/files/CODEBRIM_original_images.zip/content"
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rProgress: {percent:.1f}% ({downloaded/1e9:.2f}/{total_size/1e9:.2f} GB)", end='')
            
            urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
            print(f"\n✓ Downloaded to {zip_path}")
        
        # Extract
        print("\nExtracting CODEBRIM...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(codebrim_dir)
        print(f"✓ Extracted to {codebrim_dir}")
        
        return codebrim_dir
    
    def convert_codebrim_to_yolo(self, codebrim_dir: Path) -> Tuple[Path, int]:
        """Convert CODEBRIM XML annotations to YOLO format"""
        
        # Find images directory
        images_dir = None
        for subdir in codebrim_dir.rglob('images'):
            if subdir.is_dir():
                images_dir = subdir
                break
        
        if not images_dir:
            raise ValueError("Could not find images directory in CODEBRIM")
        
        # Find metadata
        metadata_dir = codebrim_dir / 'metadata'
        defects_xml = metadata_dir / 'defects.xml'
        
        if not defects_xml.exists():
            raise ValueError(f"defects.xml not found at {defects_xml}")
        
        # Parse XML
        print(f"Parsing {defects_xml}...")
        tree = ET.parse(defects_xml)
        root = tree.getroot()
        
        # CODEBRIM class names
        codebrim_classes = ['background', 'crack', 'spalling', 'efflorescence', 'corrosion', 'exposed_bars']
        class_mapping = self.config['CLASS_MAPPING']['codebrim']
        
        # Setup output
        output_dir = self.datasets_dir / 'codebrim_yolo'
        output_images = output_dir / 'images'
        output_labels = output_dir / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Process images
        converted_count = 0
        skipped_count = 0
        
        image_elements = root.findall('.//image')
        print(f"Found {len(image_elements)} images in XML")
        
        for image in tqdm(image_elements, desc="Converting CODEBRIM"):
            filename = image.get('filename')
            if not filename:
                continue
            
            img_path = images_dir / filename
            if not img_path.exists():
                skipped_count += 1
                continue
            
            # Get dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Copy image
            new_img_name = f"codebrim_{filename}"
            shutil.copy2(img_path, output_images / new_img_name)
            
            # Convert boxes
            yolo_lines = []
            
            for defect in image.findall('defect'):
                # Get defect classes
                defect_classes = []
                for cls_name in codebrim_classes:
                    elem = defect.find(cls_name)
                    if elem is not None and elem.text == '1':
                        defect_classes.append(cls_name)
                
                # Process boxes
                for box in defect.findall('box'):
                    for cls_name in defect_classes:
                        if cls_name in class_mapping:
                            target_class = class_mapping[cls_name]
                            if target_class in self.config['UNIFIED_CLASSES']:
                                class_id = self.config['UNIFIED_CLASSES'].index(target_class)
                                
                                # Get coordinates
                                x_min = float(box.get('xtl', 0))
                                y_min = float(box.get('ytl', 0))
                                x_max = float(box.get('xbr', 0))
                                y_max = float(box.get('ybr', 0))
                                
                                # Convert to YOLO format
                                x_center = ((x_min + x_max) / 2) / img_width
                                y_center = ((y_min + y_max) / 2) / img_height
                                width = (x_max - x_min) / img_width
                                height = (y_max - y_min) / img_height
                                
                                # Clamp
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                width = max(0, min(1, width))
                                height = max(0, min(1, height))
                                
                                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Write label
            label_path = output_labels / (new_img_name.rsplit('.', 1)[0] + '.txt')
            with open(label_path, 'w') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))
            
            converted_count += 1
        
        print(f"\n✓ Converted {converted_count} images")
        print(f"  Skipped: {skipped_count}")
        
        return output_dir, converted_count
    
    def download_roboflow_datasets(self) -> List[Tuple[str, Path]]:
        """Download datasets from Roboflow"""
        from roboflow import Roboflow
        
        downloaded = []
        
        for workspace_project, version, dataset_name in self.config['ROBOFLOW_DATASETS']:
            output_dir = self.datasets_dir / dataset_name
            
            if output_dir.exists() and (output_dir / 'data.yaml').exists():
                print(f"✓ {dataset_name}: Already downloaded")
            else:
                print(f"\nDownloading {dataset_name}...")
                try:
                    rf = Roboflow(api_key=self.config['ROBOFLOW_API_KEY'])
                    workspace, project = workspace_project.split('/')
                    project_obj = rf.workspace(workspace).project(project)
                    
                    if version is None:
                        versions = project_obj.versions()
                        version = versions[-1].version if versions else 1
                    
                    dataset = project_obj.version(version).download("yolov8", location=str(output_dir))
                    print(f"  ✓ Downloaded")
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    continue
            
            downloaded.append((dataset_name, output_dir))
        
        return downloaded
    
    def analyze_dataset(self, dataset_path: Path) -> Optional[Dict]:
        """Analyze a YOLO dataset"""
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            return None
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        classes = data.get('names', [])
        counts = {}
        
        for split in ['train', 'valid', 'test']:
            img_dir = dataset_path / split / 'images'
            counts[split] = len(list(img_dir.glob('*.*'))) if img_dir.exists() else 0
        
        return {
            'path': str(dataset_path),
            'classes': classes,
            'nc': len(classes),
            'counts': counts,
            'total': sum(counts.values())
        }
    
    def auto_map_class(self, class_name: str) -> Optional[str]:
        """Auto-map class name to unified class"""
        class_lower = class_name.lower()
        
        if any(x in class_lower for x in ['steel', 'metal', 'iron', 'beam', 'girder', 'bolt']):
            return 'steel_corrosion'
        
        if any(x in class_lower for x in ['rebar', 're-bar', 'bar', 'spall', 'spalling']):
            return 'rebar_corrosion'
        
        if any(x in class_lower for x in ['exposed', 'visible', 'bar exposed']):
            return 'exposed_rebar'
        
        if any(x in class_lower for x in ['stain', 'streak', 'discoloration']):
            return 'rust_stain'
        
        if any(x in class_lower for x in ['corrosion', 'rust', 'corroded']):
            return 'steel_corrosion'
        
        return None
    
    def merge_datasets(self, codebrim_yolo_dir: Path, roboflow_datasets: List[Tuple[str, Path]]) -> Path:
        """Merge all datasets into unified 4-class dataset"""
        
        if self.merged_dataset_path.exists() and (self.merged_dataset_path / 'data.yaml').exists():
            if not self.config['FORCE_RESTART']:
                print(f"✓ Merged dataset already exists")
                # Count existing
                train_count = len(list((self.merged_dataset_path / 'train' / 'images').glob('*.*')))
                val_count = len(list((self.merged_dataset_path / 'valid' / 'images').glob('*.*')))
                test_count = len(list((self.merged_dataset_path / 'test' / 'images').glob('*.*')))
                print(f"  Train: {train_count}, Val: {val_count}, Test: {test_count}")
                return self.merged_dataset_path
        
        print("Creating unified 4-class dataset...\n")
        
        # Create structure
        for split in ['train', 'valid', 'test']:
            (self.merged_dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.merged_dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        image_counter = 0
        class_counts = defaultdict(int)
        
        # Process CODEBRIM
        print("Processing CODEBRIM...")
        codebrim_images = codebrim_yolo_dir / 'images'
        codebrim_labels = codebrim_yolo_dir / 'labels'
        
        codebrim_files = sorted(list(codebrim_images.glob('*.jpg')))
        n_total = len(codebrim_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        random.seed(42)
        random.shuffle(codebrim_files)
        
        splits = {
            'train': codebrim_files[:n_train],
            'valid': codebrim_files[n_train:n_train+n_val],
            'test': codebrim_files[n_train+n_val:]
        }
        
        for split, files in splits.items():
            for img_file in tqdm(files, desc=f"CODEBRIM {split}", leave=False):
                new_name = f"img_{image_counter:06d}.jpg"
                shutil.copy2(img_file, self.merged_dataset_path / split / 'images' / new_name)
                
                lbl_file = codebrim_labels / (img_file.stem + '.txt')
                if lbl_file.exists():
                    with open(lbl_file, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(self.config['UNIFIED_CLASSES']):
                                new_lines.append(line)
                                class_counts[self.config['UNIFIED_CLASSES'][class_id]] += 1
                    
                    with open(self.merged_dataset_path / split / 'labels' / (new_name.rsplit('.', 1)[0] + '.txt'), 'w') as f:
                        f.writelines(new_lines)
                
                image_counter += 1
        
        # Process Roboflow datasets
        for dataset_name, dataset_path in roboflow_datasets:
            info = self.analyze_dataset(dataset_path)
            if not info:
                continue
            
            print(f"\nProcessing {dataset_name}...")
            
            # Build mapping
            mapping = {}
            for i, cls in enumerate(info['classes']):
                target = self.auto_map_class(cls)
                if target:
                    mapping[i] = target
            
            for split in ['train', 'valid', 'test']:
                src_img_dir = dataset_path / split / 'images'
                src_lbl_dir = dataset_path / split / 'labels'
                
                if not src_img_dir.exists():
                    continue
                
                for img_file in tqdm(list(src_img_dir.glob('*.*')), desc=f"{dataset_name} {split}", leave=False):
                    new_name = f"img_{image_counter:06d}{img_file.suffix}"
                    shutil.copy2(img_file, self.merged_dataset_path / split / 'images' / new_name)
                    
                    lbl_file = src_lbl_dir / (img_file.stem + '.txt')
                    if lbl_file.exists():
                        with open(lbl_file, 'r') as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            
                            orig_class = int(parts[0])
                            target_class = mapping.get(orig_class)
                            
                            if target_class and target_class in self.config['UNIFIED_CLASSES']:
                                new_class_id = self.config['UNIFIED_CLASSES'].index(target_class)
                                new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
                                class_counts[target_class] += 1
                        
                        with open(self.merged_dataset_path / split / 'labels' / (new_name.rsplit('.', 1)[0] + '.txt'), 'w') as f:
                            f.writelines(new_lines)
                    
                    image_counter += 1
        
        # Create data.yaml
        data_yaml = {
            'path': str(self.merged_dataset_path),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.config['UNIFIED_CLASSES']),
            'names': self.config['UNIFIED_CLASSES'],
            'description': 'Bridge corrosion detection with 4 classes'
        }
        
        with open(self.merged_dataset_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # Statistics
        print("\n=== Merged Dataset Statistics ===")
        for split in ['train', 'valid', 'test']:
            n = len(list((self.merged_dataset_path / split / 'images').glob('*.*')))
            print(f"  {split}: {n} images")
        
        print("\nClass distribution:")
        for cls in self.config['UNIFIED_CLASSES']:
            print(f"  {cls}: {class_counts[cls]} instances")
        
        return self.merged_dataset_path
    
    def train(self):
        """Main training method with resume capability"""
        
        # Check for existing run
        run_path = self.runs_dir / self.config['RUN_NAME']
        last_checkpoint = run_path / 'weights' / 'last.pt'
        best_checkpoint = run_path / 'weights' / 'best.pt'
        
        print("\n=== Training Status ===")
        print(f"Run directory: {run_path}")
        print(f"Device: {self.device.upper()}")
        
        if self.config['FORCE_RESTART']:
            print("⚠ FORCE_RESTART enabled - starting fresh")
            if run_path.exists():
                shutil.rmtree(run_path)
            resume_training = False
        elif last_checkpoint.exists():
            print(f"✓ Found checkpoint: {last_checkpoint}")
            print("→ Will resume training")
            resume_training = True
        else:
            print("No checkpoint found - starting fresh")
            resume_training = False
        
        # Load model
        if resume_training:
            print(f"\nLoading checkpoint...")
            model = YOLO(str(last_checkpoint))
        else:
            print("\nLoading pretrained YOLO11m...")
            model = YOLO('yolo11m.pt')
        
        print(f"Model loaded on device: {model.device}")
        
        # Training arguments
        training_args = {
            'data': str(self.merged_dataset_path / 'data.yaml'),
            'epochs': self.config['EPOCHS'],
            'imgsz': self.config['IMAGE_SIZE'],
            'batch': self.config['BATCH_SIZE'],
            'device': self.device,
            'patience': self.config['PATIENCE'],
            'save_period': 10,
            'project': str(self.runs_dir),
            'name': self.config['RUN_NAME'],
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'cache': False,  # Don't cache on Mac
            'cos_lr': True,
            # Augmentations
            'degrees': 15.0,
            'translate': 0.2,
            'scale': 0.5,
            'shear': 5.0,
            'mosaic': 1.0,
            'mixup': 0.2,
            'close_mosaic': 10,
        }
        
        print(f"\nTraining {self.config['EPOCHS']} epochs on {self.device.upper()}")
        print(f"Expected time: ~{self.config['EPOCHS'] * 4 // 60}-{self.config['EPOCHS'] * 6 // 60} hours")
        
        print("\n" + "="*60)
        if resume_training:
            print("RESUMING TRAINING")
            results = model.train(resume=True, **{k: v for k, v in training_args.items() if k != 'epochs'})
        else:
            print("STARTING FRESH TRAINING")
            results = model.train(**training_args)
        
        print("\n" + "="*60)
        print("Training Complete!")
        
        # Validation
        print("\nRunning validation on test set...")
        if best_checkpoint.exists():
            model = YOLO(str(best_checkpoint))
        
        metrics = model.val(
            data=str(self.merged_dataset_path / 'data.yaml'),
            split='test',
            imgsz=self.config['IMAGE_SIZE'],
            batch=self.config['BATCH_SIZE'],
            device=self.device,
        )
        
        print(f"\nmAP50: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        
        # Export
        print("\nExporting model...")
        for fmt in ['onnx', 'torchscript']:
            try:
                path = model.export(format=fmt, imgsz=self.config['IMAGE_SIZE'], dynamic=True)
                if os.path.exists(path):
                    dest = self.models_dir / f"{self.config['RUN_NAME']}.{fmt}"
                    shutil.copy2(path, dest)
                    print(f"  ✓ {fmt}: {dest}")
            except Exception as e:
                print(f"  ✗ {fmt}: {e}")
        
        if best_checkpoint.exists():
            final_model = self.models_dir / f"{self.config['RUN_NAME']}_best.pt"
            shutil.copy2(best_checkpoint, final_model)
            print(f"\n✓ Best model: {final_model}")
        
        return model
    
    def run(self):
        """Execute full pipeline"""
        print("="*60)
        print("Corrosion Detection Training - YOLO11")
        print("="*60)
        
        # Step 1: Download CODEBRIM
        print("\n[1/4] Downloading CODEBRIM...")
        codebrim_dir = self.download_codebrim()
        
        # Step 2: Convert CODEBRIM
        print("\n[2/4] Converting CODEBRIM to YOLO format...")
        codebrim_yolo_dir, _ = self.convert_codebrim_to_yolo(codebrim_dir)
        
        # Step 3: Download Roboflow datasets
        print("\n[3/4] Downloading Roboflow datasets...")
        roboflow_datasets = self.download_roboflow_datasets()
        
        # Step 4: Merge datasets
        print("\n[4/4] Merging datasets...")
        self.merge_datasets(codebrim_yolo_dir, roboflow_datasets)
        
        # Step 5: Train
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        self.train()
        
        print("\n" + "="*60)
        print("All tasks completed!")
        print("="*60)

if __name__ == '__main__':
    trainer = CorrosionTrainer(CONFIG)
    trainer.run()
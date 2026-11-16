#!/usr/bin/env python3
"""
Download corrosion datasets from Roboflow
"""

import os
import sys
import shutil
from pathlib import Path
from roboflow import Roboflow

# Roboflow API Key from the notebook
ROBOFLOW_API_KEY = "5Zmgm6NCmNWapNJMTOtk"

# Dataset configurations
DATASETS_TO_DOWNLOAD = [
    # Primary datasets for rebar/steel corrosion
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

def download_dataset(api_key, workspace_project, version, dataset_name, output_dir):
    """Download a dataset from Roboflow"""
    try:
        print(f"\nDownloading {dataset_name} from Roboflow...")
        
        rf = Roboflow(api_key=api_key)
        workspace, project = workspace_project.split('/')
        project_obj = rf.workspace(workspace).project(project)
        
        # Download in YOLOv8 format
        dataset = project_obj.version(version).download("yolov8", location=output_dir)
        
        print(f"✓ Downloaded {dataset_name} to {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {str(e)}")
        return False

def main():
    # Create datasets directory
    datasets_dir = Path('/Users/gabriel/Projects/Corrosion/datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Downloading Corrosion Datasets from Roboflow")
    print("="*60)
    
    downloaded = []
    failed = []
    
    for workspace_project, version, dataset_name in DATASETS_TO_DOWNLOAD:
        output_dir = datasets_dir / dataset_name
        
        # Check if already exists
        if output_dir.exists():
            print(f"\n{dataset_name}: Already exists, skipping...")
            downloaded.append(dataset_name)
            continue
        
        # Download dataset
        success = download_dataset(
            ROBOFLOW_API_KEY, 
            workspace_project, 
            version, 
            dataset_name, 
            str(output_dir)
        )
        
        if success:
            downloaded.append(dataset_name)
        else:
            failed.append(dataset_name)
    
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"Downloaded: {len(downloaded)}/{len(DATASETS_TO_DOWNLOAD)}")
    
    if downloaded:
        print("\n✓ Downloaded datasets:")
        for name in downloaded:
            print(f"  - {name}")
    
    if failed:
        print("\n✗ Failed datasets:")
        for name in failed:
            print(f"  - {name}")
    
    return len(failed) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

# YOLO11m Corrosion Detection - Google Colab Training Guide

## Quick Start (5 minutes to start training)

### Step 1: Open Colab
1. Go to https://colab.research.google.com
2. Upload this file: `Colab_YOLO11_Training.py`
3. Or create a new notebook and copy the code

### Step 2: Set Your Roboflow API Key
In the notebook, find this line:
```python
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"
```
Replace with your actual key from: https://app.roboflow.com/api-key

### Step 3: Run All Cells
1. Runtime → Run all (or press Ctrl+F9)
2. Wait for downloads (10-20 minutes)
3. Training starts automatically

---

## What's Better in Colab?

| Feature | MacBook M5 | Colab (Free) | Colab Pro |
|---------|-----------|--------------|-----------|
| **GPU** | MPS (Metal) | Tesla T4 | A100/V100 |
| **Training Time** | ~3 hours | ~4-6 hours | ~1-2 hours |
| **VRAM** | Shared 32GB | 15GB | 40-80GB |
| **Max Epochs** | Limited by heat | 100+ | 200+ |
| **Session Timeout** | None | 12 hours | 24 hours |
| **Dataset Size** | Limited by RAM | Larger | Unlimited |

---

## Expected Results in Colab

With full dataset (~27,000 images):
- **mAP50: 70-75%** (vs 59% on MacBook)
- **mAP50-95: 50-55%**
- **Precision: 75-80%**
- **Recall: 60-65%**

---

## Running the Training

### Option A: Upload to Colab (Recommended)
1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Select `Colab_YOLO11_Training.py`
4. Follow the steps above

### Option B: Copy Code
1. Create new notebook in Colab
2. Copy the contents of `Colab_YOLO11_Training.py`
3. Paste into Colab cells
4. Run

### Option C: Direct Link
Open this link (will prompt to open in Colab):
https://colab.research.google.com/github/[your-username]/[your-repo]/blob/main/Colab_YOLO11_Training.ipynb

---

## Training Monitoring

### In Colab:
```python
# Check live training progress
!cat /content/corrosion_workspace/runs/yolo11m_corrosion_colab/results.txt
```

### View logs:
1. Open the training log in Colab
2. Or run: `!tail -100 /content/corrosion_workspace/training.log`

### Monitor GPU usage:
```python
!nvidia-smi
```

---

## After Training

Your model will be automatically saved to Google Drive:
- **Model**: `MyDrive/YOLO_Corrosion_Training/models/yolo11m_corrosion_best.pt`
- **Results**: `MyDrive/YOLO_Corrosion_Training/results/`

---

## Tips for Best Results

1. **Use Colab Pro** ($10/month) for A100 GPU (5x faster)
2. **Don't let session timeout** - check every 2-3 hours
3. **Use all 9 datasets** - they're all included in the script
4. **Full resolution (640px)** - Colab can handle it
5. **100 epochs** - better convergence than 50

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `'batch': 8`
- Or reduce image size: `'imgsz': 480`

### "Session disconnected"
- Colab free has 12-hour limit
- Make sure "Prevent timeout" browser extension is installed
- Or upgrade to Colab Pro (24 hours)

### "Download failed"
- Check your Roboflow API key
- Make sure you have internet access in Colab

---

## Download Trained Model from Colab

After training completes:
```python
from google.colab import files
files.download('/content/corrosion_workspace/runs/yolo11m_corrosion_colab/weights/best.pt')
```

Or grab it from Google Drive directly.

---

## Comparison Summary

Your current MacBook results:
- Dataset: 2,400 images (1 dataset)
- mAP50: **59.2%**
- Time: 2.6 hours

Expected Colab results:
- Dataset: 27,000+ images (9 datasets)
- mAP50: **70-75%** 🎯
- Time: 4-6 hours (T4) or 1-2 hours (A100)

**Recommendation**: Run it in Colab for significantly better results!

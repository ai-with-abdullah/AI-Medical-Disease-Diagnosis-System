# Training Data Directory

This folder contains training datasets for ALL trainable disease models.

## Overview - What Needs Training

| Disease | Dataset Required | Size | Training Time |
|---------|------------------|------|---------------|
| Heart Disease | UCI Heart + Arrhythmia | ~150 KB | 2-3 minutes |
| Pneumonia | Kaggle Chest X-Ray | ~2 GB | 30-60 minutes |
| Skin Disease | HAM10000 | ~2.7 GB | 20-40 minutes |
| Color Blindness | NONE | N/A | No training needed |

## Complete Folder Structure

```
training_data/
│
├── heart_disease/
│   └── heart.csv                    (Download from GitHub)
│
├── arrhythmia/
│   └── arrhythmia.data              (Download from UCI)
│
├── pneumonia/
│   ├── train/
│   │   ├── NORMAL/                  (Normal X-ray images)
│   │   └── PNEUMONIA/               (Pneumonia X-ray images)
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
└── skin_disease/
    ├── HAM10000_images_part_1/      (Skin lesion images)
    ├── HAM10000_images_part_2/      (More images)
    └── HAM10000_metadata.csv        (Labels file - IMPORTANT!)
```

## Dataset Download Links

### 1. Heart Disease (Smallest - Start Here!)

**Dataset A: UCI Heart Disease**
- Link: https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv
- Size: 50 KB
- Place in: `training_data/heart_disease/heart.csv`

**Dataset B: UCI Arrhythmia**
- Link: https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data
- Size: 96 KB
- Place in: `training_data/arrhythmia/arrhythmia.data`

### 2. Pneumonia (Medium Size)

**Kaggle Chest X-Ray Dataset**
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Size: 2 GB
- Requires: Free Kaggle account
- Place in: `training_data/pneumonia/`

### 3. Skin Disease (Largest)

**HAM10000 Dataset**
- Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Size: 2.7 GB
- Requires: Free Kaggle account
- Place in: `training_data/skin_disease/`

## Training Commands

After placing datasets in correct folders, run these commands:

```bash
# 1. Heart Disease (Quick - 2-3 minutes)
python training_scripts/prepare_training_data.py
python training_scripts/train_heart_models.py

# 2. Pneumonia (30-60 minutes)
python training_scripts/train_pneumonia_models.py

# 3. Skin Disease (20-40 minutes)
python training_scripts/train_skin_model.py
```

## Detailed Instructions

See individual README files in each subfolder:
- `heart_disease/README.md`
- `arrhythmia/README.md`
- `pneumonia/README.md`
- `skin_disease/README.md`

Or see: `COMPREHENSIVE_TRAINING_GUIDE.md` (main project folder)

## Team Members

- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz

# Training Data Directory

This folder contains training datasets for ALL trainable disease models.

## Overview - What Needs Training

| Disease | Dataset Required | Size | Training Time |
|---------|------------------|------|---------------|
| Heart Disease | UCI Heart + Arrhythmia | ~150 KB | 2-3 minutes |
| Pneumonia | Kaggle Chest X-Ray | ~2 GB | 30-60 minutes |
| Skin Cancer | 6 Datasets (HAM10000, ISIC, etc.) | ~30 GB | 20-60 minutes |
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
└── skin_cancer/
    ├── ham10000/                    (HAM10000 dataset)
    │   ├── HAM10000_images_part_1/
    │   ├── HAM10000_images_part_2/
    │   └── HAM10000_metadata.csv
    ├── isic2019/                    (ISIC 2019 Challenge)
    ├── isic2020/                    (ISIC 2020 Challenge)
    ├── pad_ufes_20/                 (PAD-UFES-20 smartphone images)
    ├── melanoma_binary/             (Binary classification dataset)
    └── organized/                   (Auto-generated combined dataset)
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

### 3. Skin Cancer (Multi-Dataset Support)

**Dataset 1: HAM10000 (Recommended Start)**
- Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Size: 2.7 GB (10,015 images)
- Place in: `training_data/skin_cancer/ham10000/`

**Dataset 2: ISIC 2019 (High Accuracy)**
- Link: https://www.kaggle.com/datasets/andrewmvd/isic-2019
- Size: 9 GB (25,331 images)
- Place in: `training_data/skin_cancer/isic2019/`

**Dataset 3: ISIC 2020 (Melanoma Focus)**
- Link: https://www.kaggle.com/competitions/siim-isic-melanoma-classification
- Size: 15 GB (33,126 images)
- Place in: `training_data/skin_cancer/isic2020/`

**Dataset 4: PAD-UFES-20 (Smartphone Images)**
- Link: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
- Size: 500 MB (2,298 images)
- Place in: `training_data/skin_cancer/pad_ufes_20/`

**Note:** Download any combination - the training script auto-detects and combines all available datasets!

## Training Commands

After placing datasets in correct folders, run these commands:

```bash
# 1. Heart Disease (Quick - 2-3 minutes)
python training_scripts/prepare_training_data.py
python training_scripts/train_heart_models.py

# 2. Pneumonia (30-60 minutes)
python training_scripts/train_pneumonia_models.py

# 3. Skin Cancer (20-60 minutes)
python training_scripts/train_skin_model.py
```

## Detailed Instructions

See individual README files in each subfolder:
- `heart_disease/README.md`
- `arrhythmia/README.md`
- `pneumonia/README.md`
- `skin_cancer/README.md`

Or see: `COMPREHENSIVE_TRAINING_GUIDE.md` (main project folder)

## Team Members

- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz

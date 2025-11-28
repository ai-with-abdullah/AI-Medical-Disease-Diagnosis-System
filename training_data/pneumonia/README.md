# Pneumonia Dataset Setup

## Download Instructions

### Step 1: Go to Kaggle
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- You need a free Kaggle account to download

### Step 2: Download the Dataset
- Click "Download" button on Kaggle
- File: chest-xray-pneumonia.zip (about 2 GB)

### Step 3: Extract the ZIP File
After extracting, you should have:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

### Step 4: Copy Files Here
Copy the extracted folders to this location:
```
training_data/pneumonia/
├── train/
│   ├── NORMAL/      <- Copy normal X-ray images here
│   └── PNEUMONIA/   <- Copy pneumonia X-ray images here
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Dataset Information
- Total Images: ~5,800 chest X-ray images
- Classes: NORMAL, PNEUMONIA
- Image Format: JPEG
- Resolution: Various (will be resized to 224x224)

## After Setup
Run this command to train:
```bash
python training_scripts/train_pneumonia_models.py
```

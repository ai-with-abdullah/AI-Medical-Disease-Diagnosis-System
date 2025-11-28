# Skin Disease Dataset Setup (HAM10000)

## Download Instructions

### Step 1: Go to Kaggle
- Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- You need a free Kaggle account to download

### Step 2: Download the Dataset
- Click "Download" button on Kaggle
- File: archive.zip (about 2.7 GB)

### Step 3: Extract the ZIP File
After extracting, you should have:
```
archive/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── HAM10000_metadata.csv
└── hmnist_28_28_RGB.csv
```

### Step 4: Copy Files Here
Copy these files/folders to this location:
```
training_data/skin_disease/
├── HAM10000_images_part_1/   <- Copy this folder
├── HAM10000_images_part_2/   <- Copy this folder
└── HAM10000_metadata.csv     <- Copy this file (IMPORTANT!)
```

## Dataset Information
- Total Images: 10,015 skin lesion images
- Classes: 7 types of skin lesions
  1. nv - Melanocytic Nevus (Mole)
  2. mel - Melanoma
  3. bkl - Benign Keratosis
  4. bcc - Basal Cell Carcinoma
  5. akiec - Actinic Keratosis
  6. vasc - Vascular Lesion
  7. df - Dermatofibroma
- Image Format: JPEG
- Resolution: Various (will be resized to 224x224)

## After Setup
Run this command to train:
```bash
python training_scripts/train_skin_model.py
```

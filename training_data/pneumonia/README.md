# Pneumonia Dataset Setup Guide

## Overview

This folder should contain chest X-ray images for training pneumonia detection models.
The training script automatically detects and combines multiple datasets for higher accuracy.

---

## Available Datasets (Choose one or more)

| # | Dataset | Images | Size | Download Link |
|---|---------|--------|------|---------------|
| 1 | Kaggle Chest X-Ray | 5,863 | 2 GB | [Download](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| 2 | RSNA Pneumonia | 26,684 | 3 GB | [Download](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) |
| 3 | NIH ChestX-ray14 | 112,120 | 42 GB | [Download](https://www.kaggle.com/datasets/nih-chest-xrays/data) |
| 4 | NIH Resized (224x224) | 112,120 | 5 GB | [Download](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) |
| 5 | COVID-Pneumonia-Normal | 5,228 | 500 MB | [Download](https://data.mendeley.com/datasets/dvntn9yhd2/1) |
| 6 | Roboflow Chest X-Rays | 3,000+ | 300 MB | [Download](https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia) |
| 7 | VinDr-CXR | 18,000 | 15 GB | [Download](https://physionet.org/content/vindr-cxr/1.0.0/) |
| 8 | CheXpert (Stanford) | 224,316 | 440 GB | [Register](https://stanfordmlgroup.github.io/competitions/chexpert/) |

---

## Recommended Download Strategy

### For 90% Accuracy (Quick Start):
Download **Dataset 1 (Kaggle)** + **Dataset 5 (COVID-Pneumonia-Normal)**
- Total: ~11,000 images, ~2.5 GB
- Training time: 1-2 hours

### For 95% Accuracy (Recommended):
Download **Dataset 4 (NIH Resized 224x224)**
- Total: 112,120 images, ~5 GB (already preprocessed!)
- Training time: 4-6 hours

### For 97% Accuracy (Maximum):
Download multiple datasets and let the training script combine them automatically.

---

## Folder Structure

After downloading, organize your files like this:

```
training_data/pneumonia/
│
├── kaggle/                          <- Dataset 1: Kaggle Chest X-Ray
│   ├── train/
│   │   ├── NORMAL/
│   │   │   ├── IM-0001.jpeg
│   │   │   └── ... (1,341 images)
│   │   └── PNEUMONIA/
│   │       ├── person1_bacteria.jpeg
│   │       └── ... (3,875 images)
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── rsna/                            <- Dataset 2: RSNA Challenge
│   ├── images_png/                  (converted from DICOM)
│   │   ├── patient_001.png
│   │   └── ... (26,684 images)
│   └── stage_2_train_labels.csv
│
├── nih/                             <- Dataset 3/4: NIH ChestX-ray14
│   ├── organized/                   (after preprocessing)
│   │   ├── NORMAL/
│   │   │   └── ... images
│   │   └── PNEUMONIA/
│   │       └── ... images
│   ├── images/                      (raw images)
│   │   └── ... (112,120 images)
│   └── Data_Entry_2017.csv
│
├── covid_pneumonia_normal/          <- Dataset 5: Mendeley
│   ├── COVID/
│   │   └── ... (1,626 images)
│   ├── NORMAL/
│   │   └── ... (1,802 images)
│   └── PNEUMONIA/
│       └── ... (1,800 images)
│
├── roboflow/                        <- Dataset 6: Roboflow
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── valid/
│
├── vindr/                           <- Dataset 7: VinDr-CXR
│   ├── NORMAL/                      (after preprocessing)
│   ├── PNEUMONIA/
│   └── annotations_train.csv
│
├── chexpert/                        <- Dataset 8: CheXpert (Stanford)
│   ├── train/
│   │   └── patient00001/
│   │       └── study1/
│   │           └── view1_frontal.jpg
│   ├── valid/
│   ├── train.csv                    (labels file)
│   └── valid.csv
│   OR (after preprocessing):
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── nih_resized/                     <- NIH Resized (224x224) Alternative
    ├── NORMAL/                      (after preprocessing)
    ├── PNEUMONIA/
    └── Data_Entry_2017.csv          (labels file)
    OR:
    ├── images/
    │   └── ... (112,120 pre-resized images)
    └── Data_Entry_2017.csv
```

**Alternative folder names for NIH Resized:**
- `nih_resized/`
- `nih_224x224/`
- `nih-chest-x-ray-14-224x224-resized/`

---

## Quick Start Instructions

### Step 1: Download Kaggle Dataset (Easiest)

1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Login with free Kaggle account
3. Click "Download" button
4. Extract `chest-xray-pneumonia.zip`
5. Copy extracted folders to `training_data/pneumonia/kaggle/`

### Step 2: (Optional) Download Additional Datasets

For higher accuracy, download more datasets and place them in their respective folders.

### Step 3: Train the Models

```bash
python training_scripts/train_pneumonia_models.py
```

The script will:
1. Auto-detect all available datasets
2. Combine them for training
3. Train 3 CNN models (ResNet50, EfficientNet, MobileNet)
4. Save models to `models/weights/`

---

## Preprocessing (Automatic!)

**Preprocessing is now automatic!** The training script handles all preprocessing for you.

### What's Preprocessed Automatically:

| Dataset | Preprocessing | Done Automatically |
|---------|--------------|-------------------|
| RSNA | DICOM to PNG conversion | Yes |
| NIH ChestX-ray14 | Extract pneumonia/normal cases | Yes |
| NIH Resized 224x224 | Organize by class | Yes |
| VinDr-CXR | DICOM conversion + organization | Yes |
| CheXpert | Extract pneumonia-related cases | Yes |

### Optional: Run Preprocessing Manually

If you want to preprocess before training:
```bash
python training_scripts/preprocess_pneumonia_data.py
```

This will:
- Check all datasets for raw data
- Convert DICOM files to PNG
- Organize images into NORMAL/PNEUMONIA folders
- Balance classes for optimal training

**Note:** If you skip this step, the training script will automatically detect and preprocess raw data!

---

## Expected Accuracy

| Dataset Combination | Expected Accuracy |
|--------------------|-------------------|
| Kaggle only (5,863 images) | 85-88% |
| Kaggle + COVID-Pneumonia (11,000 images) | 88-92% |
| NIH Resized (112,120 images) | 92-95% |
| Multiple datasets combined (150K+ images) | 95-97% |

---

## Training Output

After training, these files will be created in `models/weights/`:

- `pneumonia_resnet50.h5` - ResNet50 model
- `pneumonia_efficientnet.h5` - EfficientNet model
- `pneumonia_mobilenet.h5` - MobileNet model

Restart the Streamlit app to use the trained models!

---

## Troubleshooting

### "Out of memory" error
- Reduce batch size in training script
- Use NIH Resized dataset (smaller file sizes)

### Low accuracy (< 85%)
- Add more training data
- Check class balance (should have similar NORMAL and PNEUMONIA counts)
- Increase training epochs

### DICOM files not loading
```bash
pip install pydicom
```

### Model not detected after training
- Check that .h5 files exist in `models/weights/`
- Restart the Streamlit app

---

## Need Help?

See `COMPREHENSIVE_TRAINING_GUIDE.md` in the project root for detailed instructions.

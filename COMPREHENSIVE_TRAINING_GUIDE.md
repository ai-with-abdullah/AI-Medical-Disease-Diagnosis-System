# Complete Training Guide - All Disease Models

This guide explains how to train ALL disease detection models for your AI Medical Disease Detection System.

---

## Team Members

| Roll Number | Name |
|-------------|------|
| F23BARIN1M01140 | Muhammad Abdullah |
| F23BARIN1M01131 | Muhammad Ali Yahya |
| F23BARIN1M01228 | Manahil Shouket |
| F23BARIN1M01114 | Ayman Noor |
| F23BARIN1M01225 | Tayyaba Mumtaz |

---

## Overview - What Needs Training

| Disease | Models Count | Dataset | Training Time |
|---------|-------------|---------|---------------|
| Pneumonia | 3 models (ResNet50, EfficientNet, MobileNet) | Chest X-ray Images | 30-60 minutes |
| Skin Disease | 1 model (ResNet50) | HAM10000 | 20-40 minutes |
| Heart Disease | 3 models (Generic CVD, CAD, Arrhythmia) | UCI Datasets | 2-3 minutes |
| Color Blindness | NO TRAINING NEEDED | Live Testing | N/A |

**Note:** Color Blindness uses live tests with predefined answers - no AI training required!

---

# PART 1: HEART DISEASE MODELS (Multiple Large Datasets!)

## Available Datasets Summary

| Dataset | Records | Size | Recommended For |
|---------|---------|------|-----------------|
| Cardiovascular Disease | 70,000 | 1.5 MB | **PRIMARY - Best for training** |
| Personal Key Indicators | 319,795 | 25 MB | Large-scale training |
| Heart Health Indicators | 253,680 | 20 MB | Alternative large dataset |
| Heart Disease Comprehensive | 1,190 | 100 KB | Combined 5 sources |
| Heart Failure Prediction | 918 | 50 KB | Quick testing |
| UCI Arrhythmia | 452 | 96 KB | Arrhythmia detection |

---

## Step 1.1: Download Heart Datasets

### Dataset 1: Cardiovascular Disease Dataset (RECOMMENDED - 70,000 records)
- **Link:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- **Size:** 1.5 MB (70,000 patients!)
- **Login Required:** Yes (free Kaggle account)
- **Download:** Click "Download" button, get `archive.zip`, extract `cardio_train.csv`

### Dataset 2: Personal Key Indicators of Heart Disease (319,795 records)
- **Link:** https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
- **Size:** 25 MB (319,795 patients!)
- **Download:** Get `heart_2022_with_nans.csv` or `heart_2022_no_nans.csv`

### Dataset 3: Heart Disease Health Indicators (253,680 records)
- **Link:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- **Size:** 20 MB (253,680 patients!)
- **Download:** Get `heart_disease_health_indicators_BRFSS2015.csv`

### Dataset 4: Heart Disease Comprehensive (1,190 records - Combined 5 Sources)
- **Link:** https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
- **Size:** 100 KB (Cleveland + Hungarian + Switzerland + Long Beach + Statlog)
- **Download:** Get `heart_statlog_cleveland_hungary_final.csv`

### Dataset 5: Heart Failure Prediction (918 records)
- **Link:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Size:** 50 KB
- **Download:** Get `heart.csv`

### Dataset 6: UCI Arrhythmia (452 records)
- **Link:** https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data
- **Right-click:** "Save As" to download
- **Size:** 96 KB

---

## Step 1.2: Place Heart Disease Files

Put the downloaded files in these folders:
```
training_data/
│
├── heart_disease/
│   ├── cardio_train.csv                              <- 70,000 records (PRIMARY)
│   ├── heart_2022_no_nans.csv                        <- 319,795 records (LARGE)
│   ├── heart_disease_health_indicators_BRFSS2015.csv <- 253,680 records (LARGE)
│   ├── heart_statlog_cleveland_hungary_final.csv     <- 1,190 records (Combined)
│   └── heart_failure.csv                             <- 918 records
│
└── arrhythmia/
    └── arrhythmia.data                               <- 452 records
```

**Note:** You can download ANY combination of these datasets. The training script automatically detects which files are present and uses all available data!

---

## Step 1.3: Prepare Heart Data

Open Terminal and run:
```bash
python training_scripts/prepare_training_data.py
```

The script will:
1. Detect all available datasets automatically
2. Load and combine data from all sources
3. Handle different column formats
4. Create unified training data

---

## Step 1.4: Train Heart Models

```bash
python training_scripts/train_heart_models.py
```

**Expected Results (with large datasets):**
- Generic CVD Model: **90-95% accuracy**
- CAD Model: **88-93% accuracy**
- Arrhythmia Model: **85-90% accuracy**

**Training Time:**
- With 70,000 records: 3-5 minutes
- With 300,000+ records: 10-15 minutes

**Files Created in models/weights/:**
- heart_generic_model.pkl
- heart_generic_scaler.pkl
- heart_cad_model.pkl
- heart_cad_scaler.pkl
- heart_arrhythmia_model.pkl
- heart_arrhythmia_scaler.pkl

---

# PART 2: PNEUMONIA DETECTION MODELS

## Step 2.1: Download Pneumonia Dataset

### Option A: Kaggle Chest X-Ray Dataset (Recommended)
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Size:** 2 GB
- **Login Required:** Yes (free Kaggle account)

### Option B: NIH Chest X-Ray Dataset (Larger)
- **Link:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Size:** 42 GB (very large!)

## Step 2.2: Extract and Place Pneumonia Files

After downloading, extract the ZIP file. Your folder structure should look like:
```
training_data/
└── pneumonia/
    ├── train/
    │   ├── NORMAL/
    │   │   ├── image1.jpeg
    │   │   ├── image2.jpeg
    │   │   └── ... (many images)
    │   └── PNEUMONIA/
    │       ├── image1.jpeg
    │       ├── image2.jpeg
    │       └── ... (many images)
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Step 2.3: Train Pneumonia Models

```bash
python training_scripts/train_pneumonia_models.py
```

**Training Time:** 30-60 minutes (depends on your computer)

**Expected Results:**
- ResNet50: ~95% accuracy
- EfficientNet: ~94% accuracy
- MobileNet: ~92% accuracy

**Files Created in models/weights/:**
- pneumonia_resnet50.h5
- pneumonia_efficientnet.h5
- pneumonia_mobilenet.h5

---

# PART 3: SKIN DISEASE DETECTION MODEL

## Step 3.1: Download Skin Disease Dataset

### HAM10000 Dataset (Recommended)
- **Link:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Size:** 2.7 GB
- **Classes:** 7 types of skin lesions

**Alternative:** ISIC Archive
- **Link:** https://www.isic-archive.com/
- **Size:** Various

## Step 3.2: Extract and Place Skin Disease Files

After downloading, extract the files:
```
training_data/
└── skin_disease/
    ├── HAM10000_images_part_1/
    │   ├── ISIC_0024306.jpg
    │   ├── ISIC_0024307.jpg
    │   └── ... (many images)
    ├── HAM10000_images_part_2/
    │   └── ... (more images)
    └── HAM10000_metadata.csv    <- Important! Contains labels
```

## Step 3.3: Train Skin Disease Model

```bash
python training_scripts/train_skin_model.py
```

**Training Time:** 20-40 minutes

**Expected Results:**
- ResNet50: ~93% accuracy

**Files Created in models/weights/:**
- skin_resnet50.h5

---

# QUICK REFERENCE - All Commands

## Complete Training (All Models)

Run these commands in order:

```bash
# Step 1: Navigate to project folder
cd AI-Medical-Disease-Diagnosis-System-main

# Step 2: Prepare heart disease data
python training_scripts/prepare_training_data.py

# Step 3: Train heart disease models (2-3 minutes)
python training_scripts/train_heart_models.py

# Step 4: Train pneumonia models (30-60 minutes)
python training_scripts/train_pneumonia_models.py

# Step 5: Train skin disease model (20-40 minutes)
python training_scripts/train_skin_model.py

# Step 6: Restart the app to use new models
python runcode.py
```

---

# FOLDER STRUCTURE - Complete Setup

```
training_data/
│
├── heart_disease/
│   ├── cardio_train.csv                              (70,000 records - Kaggle)
│   ├── heart_2022_no_nans.csv                        (319,795 records - Kaggle)
│   ├── heart_disease_health_indicators_BRFSS2015.csv (253,680 records - Kaggle)
│   ├── heart_statlog_cleveland_hungary_final.csv     (1,190 records - Kaggle)
│   ├── heart_failure.csv                             (918 records - Kaggle)
│   └── heart.csv                                     (303 records - UCI Original)
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
    └── HAM10000_metadata.csv        (Labels file)
```

---

# TRAINED MODELS - Final Output

After training, these files will be in `models/weights/`:

| File | Disease | Model Type |
|------|---------|------------|
| heart_generic_model.pkl | Heart Disease | Random Forest |
| heart_generic_scaler.pkl | Heart Disease | Data Scaler |
| heart_cad_model.pkl | Heart Disease | Random Forest |
| heart_cad_scaler.pkl | Heart Disease | Data Scaler |
| heart_arrhythmia_model.pkl | Heart Disease | Random Forest |
| heart_arrhythmia_scaler.pkl | Heart Disease | Data Scaler |
| pneumonia_resnet50.h5 | Pneumonia | ResNet50 CNN |
| pneumonia_efficientnet.h5 | Pneumonia | EfficientNet CNN |
| pneumonia_mobilenet.h5 | Pneumonia | MobileNet CNN |
| skin_resnet50.h5 | Skin Disease | ResNet50 CNN |

---

# TRAINING SCRIPTS REFERENCE

| Script | Purpose |
|--------|---------|
| `training_scripts/prepare_training_data.py` | Prepares heart disease CSV data |
| `training_scripts/train_heart_models.py` | Trains 3 heart disease models |
| `training_scripts/train_pneumonia_models.py` | Trains 3 pneumonia CNN models |
| `training_scripts/train_skin_model.py` | Trains skin disease CNN model |

---

# TIMELINE - How Long Each Part Takes

| Task | Time |
|------|------|
| Download heart disease datasets (all 6) | 5-10 minutes |
| Download pneumonia dataset | 10-30 minutes |
| Download skin disease dataset | 10-30 minutes |
| Train heart disease models (with 300K+ records) | 10-15 minutes |
| Train pneumonia models | 30-60 minutes |
| Train skin disease model | 20-40 minutes |
| **Total (with fast internet)** | **1.5-2.5 hours** |

**Note:** Training with larger datasets takes more time but gives significantly better accuracy!

---

# TROUBLESHOOTING

### "No module named tensorflow" error
```bash
pip install tensorflow
```

### "No module named sklearn" error
```bash
pip install scikit-learn
```

### "Out of memory" error during training
- Reduce batch size in training script (change 32 to 16 or 8)
- Close other programs to free memory

### Training takes too long
- Use a computer with GPU (much faster!)
- Reduce number of epochs in training script

### Models not loading after training
- Check that .h5 and .pkl files exist in `models/weights/`
- Restart the Streamlit app

---

# DEMO MODE vs TRAINED MODE

| Mode | When Active | Accuracy |
|------|-------------|----------|
| Demo Mode | Models not trained yet | Random (for testing only) |
| Trained Mode | After running training scripts | Real predictions |

The app automatically detects trained models and switches from demo to trained mode!

---

**Your app will be fully production-ready after completing all training steps!**

# Complete Training Guide - All Disease Models

This guide explains how to train ALL disease detection models for your AI Medical Disease Detection System.

---

## Overview - What Needs Training

| Disease | Models Count | Dataset | Training Time |
|---------|-------------|---------|---------------|
| Pneumonia | 3 models (ResNet50, EfficientNet, MobileNet) | 8 Datasets (400K+ images) | 1-3 hours |
| Skin Cancer | 1 model (ResNet50) | 6 Datasets (50K+ images) | 20-60 minutes |
| Heart Disease | 3 models (Generic CVD, CAD, Arrhythmia) | UCI Datasets | 2-3 minutes |
| Color Blindness | NO TRAINING NEEDED | Live Testing | N/A |

**Note:** Color Blindness uses live tests with predefined answers - no AI training required!

---

# PART 1: HEART DISEASE MODELS (3 Disease Types - Multiple Large Datasets!)

## The 3 Heart Disease Models

This system trains **3 specialized heart disease prediction models**:

| # | Model | Description | Icon |
|---|-------|-------------|------|
| 1 | **Generic Cardiovascular Disease** | General heart disease risk assessment (Yes/No) | ❤️ |
| 2 | **Coronary Artery Disease (CAD)** | Blockage in heart arteries - specific CAD detection | 💔 |
| 3 | **Cardiac Arrhythmia** | Irregular heartbeat detection and classification | 📈 |

---

## Available Datasets Summary

| Dataset | Records | Size | Usage |
|---------|---------|------|-------|
| Cardiovascular Disease | 70,000 | 1.5 MB | **Merged into combined training data** |
| Personal Key Indicators | 319,795 | 25 MB | Merged into combined training data |
| Heart Health Indicators | 253,680 | 20 MB | Merged into combined training data |
| Heart Disease Comprehensive | 1,190 | 100 KB | Merged into combined training data |
| Heart Failure Prediction | 918 | 50 KB | **YOUR MODEL - Merged into combined data** |
| UCI Heart Disease (Cleveland) | 303 | 50 KB | Merged into combined training data |
| UCI Arrhythmia | 452 | 96 KB | **YOUR MODEL - Arrhythmia only (separate)** |
| MIT-BIH Arrhythmia | 48 ECGs | varies | Reference only (unsupported format) |

**Training Data Usage:**
- **Generic CVD & CAD Models:** All heart disease CSVs are combined and used to train BOTH models
- **Arrhythmia Model:** Trained separately using ONLY the UCI arrhythmia.data file

---

## Step 1.1: Download Heart Datasets

### HEART DISEASE DATASETS (All merged together for Generic CVD & CAD models):

**Important:** All CSV files listed below are combined into a single training dataset used by BOTH the Generic CVD and CAD models. Download any combination - more data = better accuracy!

#### Dataset 1: Cardiovascular Disease Dataset (RECOMMENDED - 70,000 records)
- **Link:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- **Size:** 1.5 MB (70,000 patients!)
- **Login Required:** Yes (free Kaggle account)
- **Download:** Click "Download" button, get `archive.zip`, extract `cardio_train.csv`

#### Dataset 2: Personal Key Indicators of Heart Disease (319,795 records)
- **Link:** https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
- **Size:** 25 MB (319,795 patients!)
- **Download:** Get `heart_2022_with_nans.csv` or `heart_2022_no_nans.csv`

#### Dataset 3: Heart Disease Health Indicators (253,680 records)
- **Link:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- **Size:** 20 MB (253,680 patients!)
- **Download:** Get `heart_disease_health_indicators_BRFSS2015.csv`

#### More Heart Disease Datasets:

#### Dataset 4: Heart Disease Comprehensive (1,190 records - Combined 5 Sources)
- **Link:** https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
- **Size:** 100 KB (Cleveland + Hungarian + Switzerland + Long Beach + Statlog)
- **Download:** Get `heart_statlog_cleveland_hungary_final.csv`
- **Note:** Classic benchmark data - will be merged with other CSVs for training

#### Dataset 5: Heart Failure Prediction (918 records) - YOUR TRAINED MODEL
- **Link:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Size:** 50 KB
- **Download:** Get `heart.csv` and rename to `heart_failure.csv`
- **Note:** This is one of the datasets you already trained your model on! Will be merged with other CSVs for training.

#### Dataset 6: UCI Heart Disease - Cleveland (303 records)
- **Link:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Alt Link:** https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv
- **Size:** 50 KB
- **Download:** Save as `heart.csv`
- **Note:** Classic benchmark - will be merged with other CSVs for training

### FOR CARDIAC ARRHYTHMIA MODEL:

#### Dataset 7: UCI Arrhythmia (452 records) - YOUR TRAINED MODEL
- **Link:** https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data
- **Right-click:** "Save As" to download
- **Size:** 96 KB
- **Best For:** Arrhythmia Model
- **Note:** This is one of the datasets you already trained your model on!

### ADDITIONAL DATASETS (Reference Only - Require Custom Processing):

**Note:** The following datasets are listed for reference. They use different formats (ECG waveforms, imaging) and require custom preprocessing not included in the current training scripts.

#### Dataset 8: MIT-BIH Arrhythmia Database (ECG Waveforms)
- **Link:** https://physionet.org/content/mitdb/1.0.0/
- **Format:** WFDB (requires PhysioNet wfdb library)
- **Note:** Not automatically supported by current training pipeline

#### Dataset 9: CAD Research Database (Meta-dataset)
- **Link:** https://www.nature.com/articles/s41597-019-0206-3
- **Website:** www.cadataset.com
- **Contains:** 126 papers, 68 datasets (1992-2018)
- **Note:** Reference for CAD research and feature importance analysis

#### Dataset 10: CADICA Dataset (2024 - Coronary Angiography)
- **Link:** https://data.mendeley.com/datasets/p9bpx9ctcv/2
- **Format:** X-ray angiography images (requires deep learning pipeline)
- **Note:** For imaging-based CAD research, not clinical feature prediction

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

## Step 1.3: Train Heart Models (ONE COMMAND!)

Open Terminal and run:
```bash
python training_scripts/train_heart_models.py
```

**That's it!** The script automatically:
1. Detects all available datasets in training_data/heart_disease/
2. Loads and combines ALL CSV data into a unified dataset
3. Handles different column formats automatically
4. Loads arrhythmia data separately from training_data/arrhythmia/
5. Trains all 3 models with optimized settings
6. Saves trained models to models/weights/

**No code changes needed - just download datasets and run!**

**Important Note on Training Data:**
- **Generic CVD & CAD Models:** Both are trained on the same combined heart disease data (all CSV files merged together)
- **Arrhythmia Model:** Trained separately on UCI Arrhythmia dataset only

**Expected Results (with large datasets):**
- Generic CVD Model: **90-95% accuracy** (trained on combined CSVs)
- CAD Model: **88-93% accuracy** (trained on same combined CSVs as Generic)
- Arrhythmia Model: **85-90% accuracy** (trained on UCI arrhythmia data only)

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

# PART 2: PNEUMONIA DETECTION MODELS (8 Large Datasets - 400K+ Images!)

## Available Pneumonia Datasets Summary

| # | Dataset Name | Images | Size | Accuracy Potential | Difficulty |
|---|--------------|--------|------|-------------------|------------|
| 1 | Kaggle Chest X-Ray (Guangzhou) | 5,863 | 2 GB | 85-90% | Easy |
| 2 | RSNA Pneumonia Detection | 26,684 | 3 GB | 90-95% | Medium |
| 3 | NIH ChestX-ray14 | 112,120 | 42 GB | 92-96% | Hard |
| 4 | CheXpert (Stanford) | 224,316 | 440 GB | 94-97% | Hard |
| 5 | VinDr-CXR (Vietnamese) | 18,000 | 15 GB | 90-94% | Medium |
| 6 | COVID-Pneumonia-Normal | 5,228 | 500 MB | 88-92% | Easy |
| 7 | NIH Chest X-ray 14 (Resized 224x224) | 112,120 | 5 GB | 90-95% | Easy |
| 8 | Roboflow Chest X-Rays | 3,000+ | 300 MB | 85-90% | Easy |

**RECOMMENDED COMBINATION FOR 90%+ ACCURACY:**
- Dataset 1 (Kaggle) + Dataset 2 (RSNA) + Dataset 6 (COVID-Pneumonia) = **~37,000 images**
- OR Dataset 7 (NIH Resized) alone = **112,120 images (pre-resized to 224x224!)**

---

## Step 2.1: Download Pneumonia Datasets

### Dataset 1: Kaggle Chest X-Ray (Guangzhou) - EASIEST TO START
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Size:** 2 GB (5,863 images)
- **Classes:** NORMAL, PNEUMONIA (binary)
- **Quality:** Expert-graded by physicians
- **Login Required:** Yes (free Kaggle account)
- **Best For:** Quick start, baseline model

### Dataset 2: RSNA Pneumonia Detection Challenge - HIGHLY RECOMMENDED
- **Link:** https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- **Size:** 3 GB (26,684 images)
- **Classes:** Normal, Lung Opacity (Pneumonia), No Lung Opacity/Not Normal
- **Quality:** Labeled by 18 radiologists from 16 institutions
- **Format:** DICOM (needs conversion to JPEG/PNG)
- **Best For:** High accuracy, professional-grade annotations

### Dataset 3: NIH ChestX-ray14 - LARGEST FREE DATASET
- **Link:** https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Alternative:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Size:** 42 GB (112,120 images from 30,805 patients)
- **Classes:** 14 diseases including Pneumonia
- **Quality:** Labels from radiology reports (90%+ accuracy)
- **Format:** 1024x1024 PNG
- **Best For:** Production-level accuracy, research

### Dataset 4: CheXpert (Stanford) - PROFESSIONAL GRADE
- **Link:** https://stanfordmlgroup.github.io/competitions/chexpert/
- **Registration:** https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- **Size:** 440 GB (224,316 images from 65,240 patients)
- **Classes:** 14 observations including pneumonia-related findings
- **Quality:** Stanford Hospital data, uncertainty labels
- **Requirements:** Registration and data use agreement
- **Best For:** Research, highest accuracy potential

### Dataset 5: VinDr-CXR (Vietnamese) - HIGH QUALITY ANNOTATIONS
- **Link:** https://physionet.org/content/vindr-cxr/1.0.0/
- **GitHub:** https://github.com/vinbigdata-medical/vindr-cxr
- **Size:** 15 GB (18,000 images)
- **Classes:** 22 local labels + 6 global labels including Pneumonia
- **Quality:** Labeled by 17 radiologists (8+ years experience)
- **Format:** DICOM
- **Requirements:** PhysioNet account + CITI training
- **Best For:** Bounding box localization, multi-label classification

### Dataset 6: COVID-Pneumonia-Normal (Mendeley) - BALANCED DATASET
- **Link:** https://data.mendeley.com/datasets/dvntn9yhd2/1
- **Size:** 500 MB (5,228 images)
- **Classes:** COVID (1,626), Normal (1,802), Pneumonia (1,800)
- **Quality:** Pre-processed, resized to 256x256 PNG
- **Best For:** Quick training, balanced classes, COVID vs Pneumonia distinction

### Dataset 7: NIH Chest X-ray 14 (RESIZED 224x224) - EASIEST LARGE DATASET
- **Link:** https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized
- **Size:** 5 GB (112,120 images - already resized!)
- **Classes:** 14 diseases including Pneumonia
- **Quality:** Same as NIH original but pre-processed
- **Best For:** Fast training without preprocessing, large scale

### Dataset 8: Roboflow Chest X-Rays (Augmented)
- **Link:** https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia
- **Size:** 300 MB (~3,000+ augmented images)
- **Classes:** Normal, Pneumonia
- **Quality:** Augmented versions for better generalization
- **Best For:** Data augmentation, quick experiments

---

## Step 2.2: Recommended Download Strategies

### Strategy A: Quick Start (90%+ Accuracy) - 30 minutes download
Download these 3 datasets:
1. Dataset 1 (Kaggle Chest X-Ray) - 5,863 images
2. Dataset 6 (COVID-Pneumonia-Normal) - 5,228 images
3. Dataset 8 (Roboflow) - 3,000 images
**Total: ~14,000 images, ~3 GB**

### Strategy B: Best Accuracy (93%+ Accuracy) - 1-2 hours download
Download these datasets:
1. Dataset 2 (RSNA) - 26,684 images
2. Dataset 6 (COVID-Pneumonia-Normal) - 5,228 images
**Total: ~32,000 images, ~3.5 GB**

### Strategy C: Maximum Accuracy (95%+ Accuracy) - 3-4 hours download
Download:
1. Dataset 7 (NIH Resized 224x224) - 112,120 images (already preprocessed!)
**Total: 112,120 images, ~5 GB**

### Strategy D: Research Grade (96%+ Accuracy) - Full day download
Download all datasets and combine them.
**Total: 400,000+ images, ~60 GB**

---

## Step 2.3: Extract and Place Pneumonia Files

After downloading, organize files in this structure:

```
training_data/
└── pneumonia/
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
    │   ├── stage_2_train_images/
    │   │   ├── 00000001.dcm             (or converted .png)
    │   │   └── ... (26,684 images)
    │   └── stage_2_train_labels.csv
    │
    ├── nih/                             <- Dataset 3 or 7: NIH ChestX-ray14
    │   ├── images/
    │   │   ├── 00000001_000.png
    │   │   └── ... (112,120 images)
    │   └── Data_Entry_2017.csv          (labels file)
    │
    ├── chexpert/                        <- Dataset 4: CheXpert (Stanford)
    │   ├── train/
    │   │   └── patient00001/
    │   │       └── study1/
    │   │           └── view1_frontal.jpg
    │   ├── valid/
    │   ├── train.csv
    │   └── valid.csv
    │
    ├── vindr/                           <- Dataset 5: VinDr-CXR
    │   ├── train/
    │   │   └── *.dicom
    │   ├── test/
    │   ├── annotations_train.csv
    │   └── image_labels_train.csv
    │
    ├── covid_pneumonia_normal/          <- Dataset 6: Mendeley
    │   ├── COVID/
    │   │   └── ... (1,626 images)
    │   ├── NORMAL/
    │   │   └── ... (1,802 images)
    │   └── PNEUMONIA/
    │       └── ... (1,800 images)
    │
    └── roboflow/                        <- Dataset 8: Roboflow
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── valid/
            ├── NORMAL/
            └── PNEUMONIA/
```

---

## Step 2.4: Preprocessing (Automatic!)

**Good news:** Preprocessing is now fully automatic! The training script will automatically:
- Convert RSNA DICOM files to PNG
- Extract pneumonia and normal cases from NIH datasets
- Organize VinDr-CXR and CheXpert data
- Balance classes for optimal training

**Optional: Run preprocessing manually (before training):**
```bash
python training_scripts/preprocess_pneumonia_data.py
```

This preprocessing script handles:
- **RSNA Dataset:** DICOM to PNG conversion
- **NIH ChestX-ray14:** Extracts pneumonia and normal cases, balances classes
- **NIH Resized 224x224:** Organizes into NORMAL/PNEUMONIA folders
- **VinDr-CXR:** DICOM conversion and label-based organization
- **CheXpert:** Extracts pneumonia-related and normal cases

**Note:** If you skip this step, the training script will automatically detect raw data and preprocess it before training!

---

## Step 2.5: Train Pneumonia Models

```bash
python training_scripts/train_pneumonia_models.py
```

The training script automatically detects which datasets are present and combines them!

**Training Time (depends on data size):**
| Data Size | Training Time | Expected Accuracy |
|-----------|--------------|-------------------|
| ~5,000 images | 30-60 minutes | 85-88% |
| ~15,000 images | 1-2 hours | 88-92% |
| ~35,000 images | 2-3 hours | 92-95% |
| ~100,000+ images | 4-8 hours | 95-97% |

**Expected Results (with combined datasets):**
- ResNet50: **93-96% accuracy**
- EfficientNet: **92-95% accuracy**
- MobileNet: **90-93% accuracy**
- **Ensemble (All 3):** **94-97% accuracy**

**Files Created in models/weights/:**
- pneumonia_resnet50.h5
- pneumonia_efficientnet.h5
- pneumonia_mobilenet.h5

---

## Step 2.6: Training Features (Built-in!)

The training script (`training_scripts/train_pneumonia_models.py`) includes all best practices automatically:

### Built-in Features:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Data Augmentation** | Rotation (20°), shifts (20%), flip, zoom (20%) | Prevents overfitting |
| **Class Weights** | Auto-calculated balanced weights | Handles imbalanced data |
| **Learning Rate Scheduling** | ReduceLROnPlateau (factor=0.5, patience=2) | Optimal convergence |
| **Early Stopping** | Patience=5, restores best weights | Prevents overtraining |
| **Fine-tuning** | Unfreezes last 20 layers in Phase 2 | Maximum accuracy |
| **Auto Batch Size** | Adjusts based on dataset size | Optimal memory usage |
| **Multi-phase Training** | Phase 1: frozen base, Phase 2: fine-tune | Better transfer learning |

### Training Phases:
1. **Phase 1:** Train with frozen base model (ImageNet weights)
2. **Phase 2:** Fine-tune last 20 layers with lower learning rate (1e-5)

**No manual configuration needed** - just run the training command!

---

# PART 3: SKIN CANCER DETECTION MODEL (6 Datasets - 50K+ Images!)

## Available Skin Cancer Datasets Summary

| # | Dataset Name | Images | Size | Classes | Accuracy Potential | Difficulty |
|---|--------------|--------|------|---------|-------------------|------------|
| 1 | HAM10000 | 10,015 | 2.7 GB | 7 lesion types | 88-93% | Easy |
| 2 | ISIC 2019 Challenge | 25,331 | 9 GB | 8 lesion types | 90-95% | Medium |
| 3 | ISIC 2020 Challenge | 33,126 | 15 GB | 2 (Benign/Malignant) | 85-92% | Medium |
| 4 | PAD-UFES-20 | 2,298 | 500 MB | 6 lesion types | 85-90% | Easy |
| 5 | Melanoma Skin Cancer Dataset | 10,605 | 3 GB | 2 (Benign/Malignant) | 88-93% | Easy |
| 6 | Skin Cancer MNIST (Balanced) | 10,015 | 300 MB | 7 lesion types | 85-90% | Easy |

**RECOMMENDED COMBINATION FOR 93%+ ACCURACY:**
- Dataset 1 (HAM10000) + Dataset 2 (ISIC 2019) + Dataset 4 (PAD-UFES-20) = **~37,600 images**

---

## Step 3.1: Download Skin Cancer Datasets

### Dataset 1: HAM10000 (RECOMMENDED - Easy Start)
| Detail | Value |
|--------|-------|
| **Link** | https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 |
| **Size** | 2.7 GB (10,015 dermoscopy images) |
| **Classes** | 7 types of skin lesions |
| **Quality** | Expert-labeled by dermatologists |
| **Login Required** | Yes (free Kaggle account) |
| **Best For** | Quick start, baseline model |

### Dataset 2: ISIC 2019 Challenge (HIGH ACCURACY)
| Detail | Value |
|--------|-------|
| **Link** | https://challenge.isic-archive.com/data/#2019 |
| **Alt Link** | https://www.kaggle.com/datasets/andrewmvd/isic-2019 |
| **Size** | 9 GB (25,331 images) |
| **Classes** | 8 diagnostic categories |
| **Quality** | Competition-grade, expert annotations |
| **Best For** | High accuracy, comprehensive classification |

### Dataset 3: ISIC 2020 Challenge (MELANOMA FOCUS)
| Detail | Value |
|--------|-------|
| **Link** | https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data |
| **Size** | 15 GB (33,126 images from 2,056 patients) |
| **Classes** | Binary (Benign vs Malignant Melanoma) |
| **Quality** | Triple-reviewed by dermatologists |
| **Best For** | Melanoma-specific detection |

### Dataset 4: PAD-UFES-20 (SMARTPHONE IMAGES)
| Detail | Value |
|--------|-------|
| **Link** | https://www.kaggle.com/datasets/mahdavi1202/skin-cancer |
| **Alt Link** | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| **Size** | 500 MB (2,298 images) |
| **Classes** | 6 skin lesion types |
| **Quality** | Real-world smartphone images |
| **Best For** | Mobile app deployment, real-world scenarios |

### Dataset 5: Melanoma Skin Cancer Dataset (BINARY)
| Detail | Value |
|--------|-------|
| **Link** | https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images |
| **Size** | 3 GB (10,605 images) |
| **Classes** | 2 (Benign, Malignant) |
| **Quality** | Pre-organized, balanced classes |
| **Best For** | Binary classification, quick training |

### Dataset 6: Skin Cancer MNIST (HAM10000 Resized)
| Detail | Value |
|--------|-------|
| **Link** | https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 |
| **Size** | 300 MB (28x28 and 75x75 versions) |
| **Classes** | 7 lesion types |
| **Quality** | Pre-processed, fast loading |
| **Best For** | Quick experiments, low memory systems |

---

## Step 3.2: Recommended Download Strategies

### Strategy A: Quick Start (88-92% Accuracy) - 30 minutes download
Download these datasets:
1. Dataset 1 (HAM10000) - 10,015 images
**Total: ~10,000 images, ~2.7 GB**

### Strategy B: Best Accuracy (92-95% Accuracy) - 1-2 hours download
Download these datasets:
1. Dataset 1 (HAM10000) - 10,015 images
2. Dataset 2 (ISIC 2019) - 25,331 images
3. Dataset 4 (PAD-UFES-20) - 2,298 images
**Total: ~37,600 images, ~12 GB**

### Strategy C: Maximum Accuracy (94-97% Accuracy) - 3-4 hours download
Download all datasets and combine them.
**Total: 80,000+ images, ~30 GB**

---

## Step 3.3: Extract and Place Skin Cancer Files

After downloading, organize files in this structure:

```
training_data/skin_cancer/
|
+-- ham10000/                          <- Dataset 1: HAM10000
|   +-- HAM10000_images_part_1/
|   |   +-- ISIC_0024306.jpg
|   |   +-- ... (5,000+ images)
|   +-- HAM10000_images_part_2/
|   |   +-- ... (5,000+ images)
|   +-- HAM10000_metadata.csv          <- Labels file (REQUIRED!)
|
+-- isic2019/                          <- Dataset 2: ISIC 2019
|   +-- ISIC_2019_Training_Input/
|   |   +-- ISIC_0000000.jpg
|   |   +-- ... (25,331 images)
|   +-- ISIC_2019_Training_GroundTruth.csv
|
+-- isic2020/                          <- Dataset 3: ISIC 2020
|   +-- train/
|   |   +-- *.jpg (33,126 images)
|   +-- train.csv
|
+-- pad_ufes_20/                       <- Dataset 4: PAD-UFES-20
|   +-- images/
|   |   +-- *.png (2,298 images)
|   +-- metadata.csv
|
+-- melanoma_binary/                   <- Dataset 5: Melanoma Binary
|   +-- benign/
|   |   +-- ... (benign images)
|   +-- malignant/
|       +-- ... (malignant images)
|
+-- organized/                         <- Auto-generated: Combined dataset
    +-- nv/
    +-- mel/
    +-- bkl/
    +-- bcc/
    +-- akiec/
    +-- vasc/
    +-- df/
```

**Note:** You can download ANY combination of these datasets. The training script automatically detects which datasets are present and combines all available data!

---

## Step 3.4: Train Skin Cancer Model

```bash
python training_scripts/train_skin_model.py
```

**That's it!** The script automatically:
1. Detects all available datasets in training_data/skin_cancer/
2. Loads and combines ALL image data into a unified dataset
3. Handles different folder structures automatically
4. Applies class balancing for imbalanced data
5. Trains ResNet50 model with optimized settings
6. Saves trained model to models/weights/

**No code changes needed - just download datasets and run!**

**Training Time (depends on data size):**
| Data Size | Training Time | Expected Accuracy |
|-----------|--------------|-------------------|
| ~10,000 images | 20-40 minutes | 88-92% |
| ~25,000 images | 40-60 minutes | 90-94% |
| ~40,000 images | 1-2 hours | 93-96% |
| ~80,000+ images | 2-4 hours | 95-97% |

---

## Step 3.5: Skin Cancer Classes

The model classifies 7 types of skin lesions (using HAM10000 classification):

| Code | Disease Name | Category | Risk Level |
|------|-------------|----------|------------|
| nv | Melanocytic Nevus (Mole) | Benign | Low |
| mel | Melanoma | Malignant | HIGH - Urgent |
| bkl | Benign Keratosis | Benign | Low |
| bcc | Basal Cell Carcinoma | Malignant | Medium-High |
| akiec | Actinic Keratosis | Pre-cancerous | Medium |
| vasc | Vascular Lesion | Vascular | Low |
| df | Dermatofibroma | Benign | Low |

---

## Step 3.6: Training Features (Built-in!)

The training script includes all best practices automatically:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Multi-Dataset Support** | Auto-detects HAM10000, ISIC, PAD-UFES-20 | More data = better accuracy |
| **Data Augmentation** | Rotation, flip, zoom, shift | Prevents overfitting |
| **Class Weights** | Auto-calculated balanced weights | Handles imbalanced data |
| **Learning Rate Scheduling** | ReduceLROnPlateau | Optimal convergence |
| **Early Stopping** | Patience=5, restores best | Prevents overtraining |
| **Fine-tuning** | 2-phase transfer learning | Maximum accuracy |
| **Auto Batch Size** | Adjusts based on dataset | Optimal memory usage |

---

## Step 3.7: Files Created

After training, these files are saved to `models/weights/`:
- `skin_resnet50.h5` - Trained model
- `skin_classes.json` - Class mapping info

---

# QUICK REFERENCE - All Commands

## Complete Training (All Models)

Run these commands in order:

```bash
# Step 1: Navigate to project folder
cd AI-Medical-Disease-Diagnosis-System-main

# Step 2: Train heart disease models (2-3 minutes) - ONE COMMAND!
python training_scripts/train_heart_models.py

# Step 3: Train pneumonia models (1-3 hours depending on data)
python training_scripts/train_pneumonia_models.py

# Step 4: Train skin cancer model (20-60 minutes)
python training_scripts/train_skin_model.py

# Step 5: Restart the app to use new models
streamlit run app.py
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
│   └── arrhythmia.data                               (452 records - UCI)
│
├── pneumonia/
│   ├── kaggle/                                       (5,863 images - Kaggle)
│   │   ├── train/NORMAL/
│   │   ├── train/PNEUMONIA/
│   │   ├── val/NORMAL/
│   │   ├── val/PNEUMONIA/
│   │   └── test/
│   │
│   ├── rsna/                                         (26,684 images - RSNA)
│   │   ├── images_png/
│   │   └── stage_2_train_labels.csv
│   │
│   ├── nih/                                          (112,120 images - NIH)
│   │   ├── images/ OR organized/NORMAL/ + PNEUMONIA/
│   │   └── Data_Entry_2017.csv
│   │
│   ├── covid_pneumonia_normal/                       (5,228 images - Mendeley)
│   │   ├── COVID/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   │
│   └── roboflow/                                     (3,000+ images - Roboflow)
│       ├── train/NORMAL/
│       ├── train/PNEUMONIA/
│       └── valid/
│
└── skin_disease/
    ├── HAM10000_images_part_1/                       (Skin lesion images)
    ├── HAM10000_images_part_2/                       (More images)
    └── HAM10000_metadata.csv                         (Labels file)
```

---

# TRAINED MODELS - Final Output

After training, these files will be in `models/weights/`:

| File | Disease | Model Type | Expected Accuracy |
|------|---------|------------|-------------------|
| heart_generic_model.pkl | Heart Disease | Random Forest | 90-95% |
| heart_generic_scaler.pkl | Heart Disease | Data Scaler | - |
| heart_cad_model.pkl | Heart Disease | Random Forest | 88-93% |
| heart_cad_scaler.pkl | Heart Disease | Data Scaler | - |
| heart_arrhythmia_model.pkl | Heart Disease | Random Forest | 85-90% |
| heart_arrhythmia_scaler.pkl | Heart Disease | Data Scaler | - |
| pneumonia_resnet50.h5 | Pneumonia | ResNet50 CNN | 93-96% |
| pneumonia_efficientnet.h5 | Pneumonia | EfficientNet CNN | 92-95% |
| pneumonia_mobilenet.h5 | Pneumonia | MobileNet CNN | 90-93% |
| skin_resnet50.h5 | Skin Disease | ResNet50 CNN | 90-93% |

---

# TRAINING SCRIPTS REFERENCE

| Script | Purpose |
|--------|---------|
| `training_scripts/train_heart_models.py` | **ALL-IN-ONE:** Auto-detects datasets, prepares data, trains all 3 heart models |
| `training_scripts/train_pneumonia_models.py` | Trains 3 pneumonia CNN models (auto-detects datasets) |
| `training_scripts/train_skin_model.py` | Trains skin disease CNN model |
| `training_scripts/prepare_training_data.py` | (Optional) Advanced data preparation for all models |

---

# TIMELINE - How Long Each Part Takes

| Task | Time |
|------|------|
| Download pneumonia datasets (Quick Start - 3 datasets) | 30-60 minutes |
| Download pneumonia datasets (Full - all 8 datasets) | 3-6 hours |
| Download heart disease datasets (all 6) | 5-10 minutes |
| Download skin disease dataset | 10-30 minutes |
| Train heart disease models | 10-15 minutes |
| Train pneumonia models (15K images) | 1-2 hours |
| Train pneumonia models (100K+ images) | 4-8 hours |
| Train skin disease model | 20-40 minutes |
| **Total (Quick Start)** | **3-4 hours** |
| **Total (Full Setup)** | **8-12 hours** |

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
- Use `mixed_float16` precision for faster training with less memory

### Training takes too long
- Use a computer with GPU (much faster!)
- Reduce number of epochs in training script
- Start with smaller datasets first

### Low accuracy (below 85%)
- Add more training data (combine multiple datasets)
- Increase number of epochs
- Use data augmentation
- Try fine-tuning pre-trained layers
- Check for class imbalance and use class weights

### Models not loading after training
- Check that .h5 and .pkl files exist in `models/weights/`
- Restart the Streamlit app
- Check file permissions

### DICOM files not loading
```bash
pip install pydicom
```

---

# DEMO MODE vs TRAINED MODE

| Mode | When Active | Accuracy |
|------|-------------|----------|
| Demo Mode | Models not trained yet | Random (for testing only) |
| Trained Mode | After running training scripts | Real predictions (90%+) |

The app automatically detects trained models and switches from demo to trained mode!

---

# RECOMMENDED SETUP FOR 90%+ ACCURACY

## Minimum Requirements (90% Accuracy):
1. Download Kaggle Chest X-Ray dataset (5,863 images)
2. Download COVID-Pneumonia-Normal dataset (5,228 images)
3. Train with both combined (~11,000 images)

## Optimal Requirements (95% Accuracy):
1. Download NIH Resized 224x224 dataset (112,120 images)
2. Train with full dataset
3. Use GPU for faster training

## Maximum Requirements (97% Accuracy):
1. Download multiple datasets (RSNA + NIH + VinDr)
2. Combine all datasets (~150,000+ images)
3. Fine-tune pre-trained layers
4. Use ensemble of all 3 models

---

**Your app will achieve 90%+ accuracy after downloading multiple datasets and completing all training steps!**

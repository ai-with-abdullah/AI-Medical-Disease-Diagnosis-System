# Arrhythmia Datasets

## Overview

This folder contains datasets for **Cardiac Arrhythmia (Abnormal Heartbeat)** detection - one of the 3 heart disease prediction models in this system.

**Model Type:** Cardiac Arrhythmia Detection  
**Prediction:** Normal heartbeat vs Abnormal heartbeat patterns  

---

## Dataset 1: UCI Arrhythmia Dataset (YOUR TRAINED MODEL)

**Records:** 452 patients  
**Size:** 96 KB  
**Features:** 279 attributes (ECG measurements)  
**Classes:** 16 arrhythmia types (we use binary: Normal vs Arrhythmia)  
**Link:** https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data  

**Note:** This is one of the datasets you already trained your model on!

### Download Instructions:
1. Go to the link above
2. Right-click on `arrhythmia.data`
3. Select "Save As" and download to this folder

### Dataset Information:
- Contains ECG measurements for arrhythmia classification
- 279 features total (we use first 9 for consistency with other heart models)
- Target classes:
  - Class 1 = Normal (healthy heartbeat)
  - Classes 2-16 = Various arrhythmia types

### Feature Groups:
| Features | Description |
|----------|-------------|
| 1-15 | Patient demographics and general info |
| 16-27 | QRS wave measurements |
| 28-159 | DII lead measurements |
| 160-279 | V1-V6 lead measurements |

---

## Additional ECG Datasets (Reference Only)

**Note:** The following ECG datasets are listed for reference and future research. The current training pipeline (`prepare_training_data.py`) is designed to work with the UCI Arrhythmia dataset format. These ECG datasets require different preprocessing and are not automatically supported by the current scripts.

### Dataset 2: MIT-BIH Arrhythmia Database

**Records:** 48 half-hour ECG excerpts  
**Subjects:** 47 patients  
**Sampling Rate:** 360 Hz  
**Link:** https://physionet.org/content/mitdb/1.0.0/  
**Format:** WFDB (requires separate processing)

### Dataset 3: PTB Diagnostic ECG Database

**Records:** 549 records  
**Subjects:** 290 patients  
**Link:** https://physionet.org/content/ptbdb/1.0.0/  
**Format:** WFDB (requires separate processing)

### Dataset 4: CPSC 2018 ECG Dataset

**Records:** 6,877 12-lead ECG recordings  
**Link:** http://2018.icbeb.org/Challenge.html  
**Format:** MAT files (requires separate processing)

---

## Expected Files in This Folder

After downloading, you should have:
```
training_data/arrhythmia/
├── arrhythmia.data          (452 records - UCI - YOUR MODEL)
├── README.md                (This file)
└── (optional) mitdb/        (MIT-BIH if downloaded)
```

---

## Quick Start

**Minimum Required:** Just download `arrhythmia.data` from UCI repository.

```bash
# Direct download command
curl -o training_data/arrhythmia/arrhythmia.data https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data
```

---

## Arrhythmia Types Detected

The UCI dataset classifies these arrhythmia types:

| Class | Arrhythmia Type |
|-------|-----------------|
| 1 | Normal |
| 2 | Ischemic changes (Coronary Artery Disease) |
| 3 | Old Anterior Myocardial Infarction |
| 4 | Old Inferior Myocardial Infarction |
| 5 | Sinus tachycardia |
| 6 | Sinus bradycardia |
| 7 | Ventricular Premature Contraction (PVC) |
| 8 | Supraventricular Premature Contraction |
| 9 | Left bundle branch block |
| 10 | Right bundle branch block |
| 11 | 1st degree AtrioVentricular block |
| 12 | 2nd degree AV block |
| 13 | 3rd degree AV block |
| 14 | Left ventricular hypertrophy |
| 15 | Atrial Fibrillation or Flutter |
| 16 | Others |

**For our model:** We use binary classification (Class 1 = Normal, Classes 2-16 = Arrhythmia)

---

## Training Script Usage

After placing the data file:
```bash
# Step 1: Prepare data
python training_scripts/prepare_training_data.py

# Step 2: Train models (includes arrhythmia)
python training_scripts/train_heart_models.py
```

---

## License
- UCI Arrhythmia: CC BY 4.0 (Free to use)
- MIT-BIH: PhysioNet License (Free for research)
- PTB: PhysioNet License (Free for research)

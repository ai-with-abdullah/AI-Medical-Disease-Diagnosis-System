# Heart Disease Datasets

## Available Datasets (Total: 700,000+ records)

This folder supports **multiple heart disease datasets** for training 3 different models:
1. **Generic Cardiovascular Disease (CVD)** - General heart disease risk prediction
2. **Coronary Artery Disease (CAD)** - Blockage in heart arteries detection
3. **Cardiac Arrhythmia** - Irregular heartbeat patterns (uses separate arrhythmia folder)

Download any combination for training - more data = better accuracy!

---

## Core Datasets (Already Supported)

**Important:** All CSV files in this folder are automatically detected and merged into a single combined training dataset. This merged data is used to train BOTH the Generic CVD and CAD models. Only the Arrhythmia model uses separate data (from the arrhythmia folder).

### Dataset 1: Cardiovascular Disease Dataset (RECOMMENDED - PRIMARY)
**Records:** 70,000 patients  
**Size:** 1.5 MB  
**Link:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

#### Download Instructions:
1. Go to the link above
2. Click "Download" button (requires free Kaggle account)
3. Extract `archive.zip`
4. Copy `cardio_train.csv` to this folder

#### Features:
| Column | Description |
|--------|-------------|
| age | Age in days (divide by 365 for years) |
| gender | 1 = female, 2 = male |
| height | Height in cm |
| weight | Weight in kg |
| ap_hi | Systolic blood pressure |
| ap_lo | Diastolic blood pressure |
| cholesterol | 1 = normal, 2 = above normal, 3 = well above |
| gluc | Glucose: 1 = normal, 2 = above, 3 = well above |
| smoke | Smoking: 0 = no, 1 = yes |
| alco | Alcohol: 0 = no, 1 = yes |
| active | Physical activity: 0 = no, 1 = yes |
| cardio | Target: 0 = no disease, 1 = disease |

---

### Dataset 2: Personal Key Indicators of Heart Disease (LARGEST)
**Records:** 319,795 patients  
**Size:** 25 MB  
**Link:** https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

#### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Extract and copy `heart_2022_no_nans.csv` to this folder

#### Features:
- HeartDisease, BMI, Smoking, AlcoholDrinking, Stroke
- PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory
- Race, Diabetic, PhysicalActivity, GenHealth, SleepTime
- Asthma, KidneyDisease, SkinCancer

---

### Dataset 3: Heart Disease Health Indicators (BRFSS 2015)
**Records:** 253,680 patients  
**Size:** 20 MB  
**Link:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

#### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Copy `heart_disease_health_indicators_BRFSS2015.csv` to this folder

#### Features:
- 21 health indicator features from CDC BRFSS survey
- Binary classification target

---

### Dataset 4: Heart Disease Comprehensive (Combined 5 Sources)
**Records:** 1,190 patients  
**Size:** 100 KB  
**Link:** https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final  
**Note:** Classic benchmark data - merged with all other CSVs for training

#### Sources Combined:
- Cleveland Clinic Foundation (303 records)
- Hungarian Institute of Cardiology (294 records)
- University Hospital Zurich (123 records)
- V.A. Medical Center Long Beach (200 records)
- Statlog Heart Dataset (270 records)

#### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Copy `heart_statlog_cleveland_hungary_final.csv` to this folder

---

### Dataset 5: Heart Failure Prediction (YOUR TRAINED MODEL)
**Records:** 918 patients  
**Size:** 50 KB  
**Link:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction  

**Note:** This is one of the datasets you already trained your model on! Will be merged with all other CSVs for training.

#### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Rename to `heart_failure.csv` and copy to this folder

#### Features:
| Column | Description |
|--------|-------------|
| Age | Age of the patient |
| Sex | M = Male, F = Female |
| ChestPainType | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y = Yes, N = No |
| HeartDisease | 1 = heart disease, 0 = Normal |

---

### Dataset 6: UCI Heart Disease (Original - Cleveland)
**Records:** 303 patients  
**Size:** 50 KB  
**Link:** https://archive.ics.uci.edu/dataset/45/heart+disease  
**Alt Link:** https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv  
**Note:** Classic benchmark - merged with all other CSVs for training

#### Download Instructions:
1. Go to the link above
2. Click "Download raw file"
3. Save as `heart.csv` in this folder

#### Python Access (Alternative):
```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
```

---

## Additional Recommended Datasets (2024 - Reference Only)

**Note:** These additional datasets are listed for reference. They may require different preprocessing and are not automatically supported by the current training pipeline.

### Dataset 7: Framingham Heart Study
**Records:** Variable (10-year CHD risk)  
**Link:** Search on Kaggle: "Framingham Heart Study"  
**Use Case:** Risk scoring, longitudinal analysis (requires custom preprocessing)

---

### Dataset 8: CAD Research Database (Meta-dataset)
**Records:** 126 papers, 68 datasets  
**Link:** https://www.nature.com/articles/s41597-019-0206-3  
**Website:** www.cadataset.com  
**Use Case:** Feature importance analysis, research reference (not for direct training)

---

### Dataset 9: Coronary Heart Disease (CHD) Dataset
**Records:** 462 instances  
**Features:** 10 features  
**Use Case:** CAD binary classification (requires custom preprocessing)

---

## Expected Files in This Folder

After downloading, you should have some or all of these files:
```
training_data/heart_disease/
├── cardio_train.csv                              (70,000 records - PRIMARY)
├── heart_2022_no_nans.csv                        (319,795 records - LARGEST)
├── heart_disease_health_indicators_BRFSS2015.csv (253,680 records)
├── heart_statlog_cleveland_hungary_final.csv     (1,190 records)
├── heart_failure.csv                             (918 records - YOUR MODEL)
└── heart.csv                                     (303 records - UCI)
```

**Note:** The training script automatically detects which files are present and uses all available data!

---

## Quick Start (Minimum Required)

For best results, download at least:
1. `cardio_train.csv` (70,000 records) - PRIMARY dataset (merged into combined data)
2. `heart_failure.csv` (918 records) - YOUR TRAINED MODEL (merged into combined data)
3. `arrhythmia.data` (put in ../arrhythmia/ folder) - For Arrhythmia model (separate)

This gives you 70,000+ records for high-accuracy training!

**Remember:** All CSV files in this folder are merged together and used to train BOTH Generic CVD and CAD models. Only the Arrhythmia model uses separate data.

---

## How Training Works

The training script (`prepare_training_data.py`) automatically:
1. **Detects all available CSV files** in this folder
2. **Combines ALL data sources** into a single unified training dataset
3. **Creates model files** for Generic CVD, CAD, and Arrhythmia

**Important:** Both the Generic CVD and CAD models are trained on the **same combined dataset** (all CSV files merged together). They use the same training data but are saved as separate model files. The Arrhythmia model is the only one trained on a separate dataset (UCI arrhythmia.data from ../arrhythmia/ folder).

| Model | Data Source | Records |
|-------|-------------|---------|
| Generic CVD | All heart disease CSVs combined | 400K+ |
| Coronary Artery Disease (CAD) | Same combined data as Generic CVD | 400K+ |
| Cardiac Arrhythmia | UCI arrhythmia.data only | 452+ |

---

## License
All datasets are free to use for research and educational purposes (CC BY 4.0)

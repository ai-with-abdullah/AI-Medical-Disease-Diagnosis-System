# Heart Disease Datasets

## Available Datasets (Total: 645,000+ records)

This folder supports **6 different heart disease datasets**. Download any combination for training - more data = better accuracy!

---

## Dataset 1: Cardiovascular Disease Dataset (RECOMMENDED)
**Records:** 70,000 patients  
**Size:** 1.5 MB  
**Link:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

### Download Instructions:
1. Go to the link above
2. Click "Download" button (requires free Kaggle account)
3. Extract `archive.zip`
4. Copy `cardio_train.csv` to this folder

### Features:
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

## Dataset 2: Personal Key Indicators of Heart Disease (LARGEST)
**Records:** 319,795 patients  
**Size:** 25 MB  
**Link:** https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Extract and copy `heart_2022_no_nans.csv` to this folder

### Features:
- HeartDisease, BMI, Smoking, AlcoholDrinking, Stroke
- PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory
- Race, Diabetic, PhysicalActivity, GenHealth, SleepTime
- Asthma, KidneyDisease, SkinCancer

---

## Dataset 3: Heart Disease Health Indicators
**Records:** 253,680 patients  
**Size:** 20 MB  
**Link:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Copy `heart_disease_health_indicators_BRFSS2015.csv` to this folder

### Features:
- 21 health indicator features from CDC BRFSS survey
- Binary classification target

---

## Dataset 4: Heart Disease Comprehensive (Combined 5 Sources)
**Records:** 1,190 patients  
**Size:** 100 KB  
**Link:** https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final

### Sources Combined:
- Cleveland Clinic Foundation (303 records)
- Hungarian Institute of Cardiology (294 records)
- University Hospital Zurich (123 records)
- V.A. Medical Center Long Beach (200 records)
- Statlog Heart Dataset (270 records)

### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Copy `heart_statlog_cleveland_hungary_final.csv` to this folder

---

## Dataset 5: Heart Failure Prediction
**Records:** 918 patients  
**Size:** 50 KB  
**Link:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

### Download Instructions:
1. Go to the link above
2. Click "Download"
3. Rename to `heart_failure.csv` and copy to this folder

---

## Dataset 6: UCI Heart Disease (Original)
**Records:** 303 patients  
**Size:** 50 KB  
**Link:** https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv

### Download Instructions:
1. Go to the link above
2. Click "Download raw file"
3. Save as `heart.csv` in this folder

---

## Expected Files in This Folder

After downloading, you should have some or all of these files:
```
training_data/heart_disease/
├── cardio_train.csv                              (70,000 records)
├── heart_2022_no_nans.csv                        (319,795 records)
├── heart_disease_health_indicators_BRFSS2015.csv (253,680 records)
├── heart_statlog_cleveland_hungary_final.csv     (1,190 records)
├── heart_failure.csv                             (918 records)
└── heart.csv                                     (303 records)
```

**Note:** The training script automatically detects which files are present and uses all available data!

---

## Quick Start (Minimum Required)

For best results, download at least:
1. `cardio_train.csv` (70,000 records) - PRIMARY dataset
2. `arrhythmia.data` (put in ../arrhythmia/ folder)

This gives you 70,000+ records for high-accuracy training!

---

## License
All datasets are free to use for research and educational purposes (CC BY 4.0)

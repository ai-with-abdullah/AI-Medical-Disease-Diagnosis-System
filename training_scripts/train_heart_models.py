"""
Train Heart Disease Models - ALL-IN-ONE SCRIPT
===============================================
Just download datasets and run this script - no code changes needed!

This script automatically:
1. Detects all available datasets in training_data/heart_disease/
2. Loads and combines them (for Generic CVD & CAD models)
3. Loads arrhythmia data separately (from training_data/arrhythmia/)
4. Trains all 3 models with optimized settings

Supported Datasets (place in training_data/heart_disease/):
-----------------------------------------------------------
1. cardio_train.csv              - Cardiovascular Disease (70,000 records)
2. heart_2022_no_nans.csv        - Personal Key Indicators (319,795 records)
3. heart_2022_with_nans.csv      - Personal Key Indicators (alt version)
4. heart_disease_health_indicators_BRFSS2015.csv - Health Indicators (253,680 records)
5. heart_statlog_cleveland_hungary_final.csv     - Comprehensive 5-source (1,190 records)
6. heart_failure.csv             - Heart Failure Prediction (918 records)
7. heart.csv                     - UCI Original Cleveland (303 records)

Arrhythmia Dataset (place in training_data/arrhythmia/):
--------------------------------------------------------
- arrhythmia.data                - UCI Arrhythmia (452 records)

Usage:
------
1. Download datasets from links in COMPREHENSIVE_TRAINING_GUIDE.md
2. Place CSV files in training_data/heart_disease/
3. Place arrhythmia.data in training_data/arrhythmia/
4. Run: python training_scripts/train_heart_models.py

That's it! No code changes required.

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_cardiovascular_dataset(filepath):
    """Load Cardiovascular Disease Dataset (70,000 records)
    File: cardio_train.csv
    Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    """
    print(f"      Loading Cardiovascular Disease Dataset...")
    df = pd.read_csv(filepath, sep=';')
    print(f"      Loaded {len(df):,} records")
    
    df['age_years'] = (df['age'] / 365.25).round(0)
    df['sex'] = df['gender'].map({1: 0, 2: 1})
    df['cp'] = 0
    df['trestbps'] = df['ap_hi']
    df['chol'] = df['cholesterol'] * 100 + 100
    df['fbs'] = (df['gluc'] > 1).astype(int)
    df['restecg'] = 0
    df['thalach'] = 150 - (df['age_years'] * 0.5).astype(int)
    df['exang'] = (df['active'] == 0).astype(int)
    
    feature_cols = ['age_years', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    X = df[feature_cols].values.astype(float)
    y = df['cardio'].values.astype(int)
    
    return X, y


def load_personal_indicators_dataset(filepath):
    """Load Personal Key Indicators of Heart Disease (319,795 records)
    Files: heart_2022_no_nans.csv OR heart_2022_with_nans.csv
    Source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
    """
    print(f"      Loading Personal Key Indicators Dataset...")
    df = pd.read_csv(filepath)
    print(f"      Loaded {len(df):,} records")
    
    age_map = {
        '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
        '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
        '70-74': 72, '75-79': 77, '80 or older': 82
    }
    
    if 'AgeCategory' in df.columns:
        df['age_years'] = df['AgeCategory'].map(age_map).fillna(50)
    elif 'Age' in df.columns:
        df['age_years'] = df['Age']
    else:
        df['age_years'] = 50
    
    if 'Sex' in df.columns:
        df['sex'] = (df['Sex'] == 'Male').astype(int)
    else:
        df['sex'] = 1
    
    df['cp'] = 0
    
    if 'BMI' in df.columns:
        df['trestbps'] = (120 + (df['BMI'] - 25) * 2).clip(80, 200)
        df['chol'] = (200 + (df['BMI'] - 25) * 5).clip(100, 400)
    else:
        df['trestbps'] = 120
        df['chol'] = 200
    
    if 'Diabetic' in df.columns:
        df['fbs'] = df['Diabetic'].apply(lambda x: 1 if str(x) in ['Yes', 'Yes (during pregnancy)', '1', '1.0'] else 0)
    else:
        df['fbs'] = 0
    
    df['restecg'] = 0
    df['thalach'] = 220 - df['age_years']
    
    if 'PhysicalActivity' in df.columns:
        df['exang'] = (df['PhysicalActivity'] == 'No').astype(int)
    else:
        df['exang'] = 0
    
    feature_cols = ['age_years', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    X = df[feature_cols].values.astype(float)
    
    if 'HeartDisease' in df.columns:
        if df['HeartDisease'].dtype == object:
            y = (df['HeartDisease'] == 'Yes').astype(int).values
        else:
            y = df['HeartDisease'].astype(int).values
    else:
        y = df.iloc[:, 0].astype(int).values
    
    return X, y


def load_health_indicators_dataset(filepath):
    """Load Heart Disease Health Indicators (253,680 records)
    File: heart_disease_health_indicators_BRFSS2015.csv
    Source: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
    """
    print(f"      Loading Health Indicators Dataset...")
    df = pd.read_csv(filepath)
    print(f"      Loaded {len(df):,} records")
    
    if 'Age' in df.columns:
        df['age_years'] = df['Age'] * 5 + 20
    else:
        df['age_years'] = 50
    
    if 'Sex' in df.columns:
        df['sex'] = df['Sex'].astype(int)
    else:
        df['sex'] = 1
    
    df['cp'] = 0
    
    if 'HighBP' in df.columns:
        df['trestbps'] = df['HighBP'].astype(int) * 40 + 100
    else:
        df['trestbps'] = 120
    
    if 'HighChol' in df.columns:
        df['chol'] = df['HighChol'].astype(int) * 100 + 150
    else:
        df['chol'] = 200
    
    if 'Diabetes' in df.columns:
        df['fbs'] = (df['Diabetes'] > 0).astype(int)
    else:
        df['fbs'] = 0
    
    df['restecg'] = 0
    df['thalach'] = 150
    
    if 'PhysActivity' in df.columns:
        df['exang'] = (df['PhysActivity'] == 0).astype(int)
    else:
        df['exang'] = 0
    
    feature_cols = ['age_years', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    X = df[feature_cols].values.astype(float)
    
    if 'HeartDiseaseorAttack' in df.columns:
        y = df['HeartDiseaseorAttack'].astype(int).values
    else:
        y = df.iloc[:, 0].astype(int).values
    
    return X, y


def load_comprehensive_dataset(filepath):
    """Load Heart Disease Comprehensive (Combined 5 sources - 1,190 records)
    File: heart_statlog_cleveland_hungary_final.csv
    Source: https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
    """
    print(f"      Loading Comprehensive Dataset (5 sources)...")
    df = pd.read_csv(filepath)
    print(f"      Loaded {len(df):,} records")
    
    col_mapping = {
        'age': 'age', 'Age': 'age',
        'sex': 'sex', 'Sex': 'sex',
        'chest pain type': 'cp', 'cp': 'cp', 'ChestPainType': 'cp',
        'resting bp s': 'trestbps', 'trestbps': 'trestbps', 'RestingBP': 'trestbps',
        'cholesterol': 'chol', 'chol': 'chol', 'Cholesterol': 'chol',
        'fasting blood sugar': 'fbs', 'fbs': 'fbs', 'FastingBS': 'fbs',
        'resting ecg': 'restecg', 'restecg': 'restecg', 'RestingECG': 'restecg',
        'max heart rate': 'thalach', 'thalach': 'thalach', 'MaxHR': 'thalach',
        'exercise angina': 'exang', 'exang': 'exang', 'ExerciseAngina': 'exang',
        'target': 'target', 'HeartDisease': 'target'
    }
    
    df = df.rename(columns=col_mapping)
    
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) < 5:
        X = df.iloc[:, :9].values.astype(float)
    else:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        X = df[feature_cols].values.astype(float)
    
    if 'target' in df.columns:
        y = (df['target'] > 0).astype(int).values
    else:
        y = (df.iloc[:, -1] > 0).astype(int).values
    
    return X, y


def load_heart_failure_dataset(filepath):
    """Load Heart Failure Prediction Dataset (918 records)
    File: heart_failure.csv
    Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
    """
    print(f"      Loading Heart Failure Dataset...")
    df = pd.read_csv(filepath)
    print(f"      Loaded {len(df):,} records")
    
    col_mapping = {
        'Age': 'age', 'age': 'age',
        'Sex': 'sex', 'sex': 'sex',
        'ChestPainType': 'cp', 'cp': 'cp',
        'RestingBP': 'trestbps', 'trestbps': 'trestbps',
        'Cholesterol': 'chol', 'chol': 'chol',
        'FastingBS': 'fbs', 'fbs': 'fbs',
        'RestingECG': 'restecg', 'restecg': 'restecg',
        'MaxHR': 'thalach', 'thalach': 'thalach',
        'ExerciseAngina': 'exang', 'exang': 'exang',
        'HeartDisease': 'target', 'target': 'target'
    }
    
    df = df.rename(columns=col_mapping)
    
    if 'sex' in df.columns and df['sex'].dtype == object:
        df['sex'] = (df['sex'] == 'M').astype(int)
    
    if 'cp' in df.columns and df['cp'].dtype == object:
        cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        df['cp'] = df['cp'].map(cp_map).fillna(0)
    
    if 'exang' in df.columns and df['exang'].dtype == object:
        df['exang'] = (df['exang'] == 'Y').astype(int)
    
    if 'restecg' in df.columns and df['restecg'].dtype == object:
        ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
        df['restecg'] = df['restecg'].map(ecg_map).fillna(0)
    
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].values.astype(float)
    
    if 'target' in df.columns:
        y = df['target'].astype(int).values
    else:
        y = df.iloc[:, -1].astype(int).values
    
    return X, y


def load_uci_original_dataset(filepath):
    """Load UCI Original Heart Disease Dataset (303 records)
    File: heart.csv
    Source: https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv
    """
    print(f"      Loading UCI Original Dataset...")
    df = pd.read_csv(filepath)
    print(f"      Loaded {len(df):,} records")
    
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        X = df.iloc[:, :9].values.astype(float)
    else:
        X = df[feature_cols].values.astype(float)
    
    target_col = df.columns[-1]
    y = (df[target_col] > 0).astype(int).values
    
    return X, y


def load_arrhythmia_dataset(filepath):
    """Load UCI Arrhythmia Dataset (452 records)
    File: arrhythmia.data
    Source: https://archive.ics.uci.edu/dataset/5/arrhythmia
    """
    print(f"      Loading Arrhythmia Dataset...")
    df = pd.read_csv(filepath, header=None, na_values='?')
    print(f"      Loaded {len(df):,} records")
    
    feature_df = df.iloc[:, :9].copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    X = feature_df.values.astype(float)
    
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = np.isnan(col)
        if mask.any():
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0.0
            col[mask] = col_mean
    
    y_raw = pd.to_numeric(df.iloc[:, -1], errors='coerce').fillna(1).values
    y = (y_raw > 1).astype(int)
    
    return X, y


def auto_detect_and_load_datasets():
    """Automatically detect and load all available heart disease datasets"""
    print("\n" + "=" * 70)
    print("STEP 1: AUTO-DETECTING DATASETS")
    print("=" * 70)
    
    heart_dir = os.path.join(project_dir, 'training_data', 'heart_disease')
    arrhythmia_dir = os.path.join(project_dir, 'training_data', 'arrhythmia')
    
    os.makedirs(heart_dir, exist_ok=True)
    os.makedirs(arrhythmia_dir, exist_ok=True)
    
    datasets = {
        'cardio_train.csv': ('Cardiovascular Disease (70K)', load_cardiovascular_dataset),
        'heart_2022_no_nans.csv': ('Personal Key Indicators (319K)', load_personal_indicators_dataset),
        'heart_2022_with_nans.csv': ('Personal Key Indicators Alt (319K)', load_personal_indicators_dataset),
        'heart_disease_health_indicators_BRFSS2015.csv': ('Health Indicators (253K)', load_health_indicators_dataset),
        'heart_statlog_cleveland_hungary_final.csv': ('Comprehensive 5-Source (1.1K)', load_comprehensive_dataset),
        'heart_failure.csv': ('Heart Failure Prediction (918)', load_heart_failure_dataset),
        'heart.csv': ('UCI Original Cleveland (303)', load_uci_original_dataset),
    }
    
    X_all = []
    y_all = []
    total_records = 0
    loaded_datasets = 0
    
    print("\n   Scanning training_data/heart_disease/ for datasets...")
    print("-" * 60)
    
    for filename, (name, loader) in datasets.items():
        filepath = os.path.join(heart_dir, filename)
        if os.path.exists(filepath):
            try:
                X, y = loader(filepath)
                X_all.append(X)
                y_all.append(y)
                total_records += len(y)
                loaded_datasets += 1
                print(f"   [OK] {name}")
            except Exception as e:
                print(f"   [ERROR] {name}: {str(e)[:50]}")
        else:
            print(f"   [--] {name}: Not found")
    
    if loaded_datasets == 0:
        print("\n" + "!" * 70)
        print("ERROR: No heart disease datasets found!")
        print("!" * 70)
        print("\nPlease download at least one dataset:")
        print("  1. cardio_train.csv (RECOMMENDED - 70K records)")
        print("     https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        print("\n  2. heart.csv (Quick start - 303 records)")
        print("     https://github.com/sharmaroshan/Heart-UCI-Dataset/raw/master/heart.csv")
        print(f"\nPlace files in: {heart_dir}")
        print("\nSee COMPREHENSIVE_TRAINING_GUIDE.md for all dataset links.")
        return None, None, None, None
    
    print("-" * 60)
    print(f"\n   Found {loaded_datasets} dataset(s) with {total_records:,} total records")
    
    print("\n   Combining datasets...")
    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)
    
    nan_mask = np.isnan(X_combined)
    if nan_mask.any():
        print("   Handling missing values...")
        for i in range(X_combined.shape[1]):
            col = X_combined[:, i]
            mask = np.isnan(col)
            if mask.any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0.0
                col[mask] = col_mean
    
    print(f"\n   Combined dataset:")
    print(f"      Shape: {X_combined.shape}")
    print(f"      Total: {total_records:,} records")
    print(f"      Healthy: {np.sum(y_combined == 0):,} ({100*np.sum(y_combined == 0)/len(y_combined):.1f}%)")
    print(f"      Disease: {np.sum(y_combined == 1):,} ({100*np.sum(y_combined == 1)/len(y_combined):.1f}%)")
    
    print("\n   Checking for Arrhythmia dataset...")
    print("-" * 60)
    
    arrhythmia_path = os.path.join(arrhythmia_dir, 'arrhythmia.data')
    X_arrhythmia = None
    y_arrhythmia = None
    
    if os.path.exists(arrhythmia_path):
        try:
            X_arrhythmia, y_arrhythmia = load_arrhythmia_dataset(arrhythmia_path)
            print(f"   [OK] Arrhythmia Dataset ({len(y_arrhythmia):,} records)")
            print(f"      Normal: {np.sum(y_arrhythmia == 0):,}")
            print(f"      Arrhythmia: {np.sum(y_arrhythmia == 1):,}")
        except Exception as e:
            print(f"   [ERROR] Arrhythmia: {str(e)[:50]}")
            print("   Using combined heart data for arrhythmia model")
            X_arrhythmia = X_combined.copy()
            y_arrhythmia = y_combined.copy()
    else:
        print("   [--] arrhythmia.data: Not found")
        print("   Using combined heart data for arrhythmia model")
        X_arrhythmia = X_combined.copy()
        y_arrhythmia = y_combined.copy()
    
    return X_combined, y_combined, X_arrhythmia, y_arrhythmia


def build_model(n_samples):
    """Build optimized Random Forest model based on dataset size"""
    
    if n_samples > 100000:
        print("      Using LARGE dataset config (100K+ records)")
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
    elif n_samples > 10000:
        print("      Using MEDIUM dataset config (10K+ records)")
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
    else:
        print("      Using SMALL dataset config (<10K records)")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )


def train_single_model(X, y, model_name):
    """Train a single model with comprehensive evaluation"""
    
    n_samples = len(y)
    print(f"      Dataset size: {n_samples:,} records")
    print(f"      Classes: {np.sum(y==0):,} healthy, {np.sum(y==1):,} disease")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"      Train: {len(y_train):,}, Test: {len(y_test):,}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = build_model(n_samples)
    
    print("      Training... (please wait)")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"      Completed in {train_time:.1f} seconds")
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    auc = 0.0
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0
    
    print(f"\n      === RESULTS ===")
    print(f"      Accuracy:  {accuracy:.2%}")
    print(f"      Precision: {precision:.2%}")
    print(f"      Recall:    {recall:.2%}")
    print(f"      F1 Score:  {f1:.2%}")
    if auc > 0:
        print(f"      ROC-AUC:   {auc:.2%}")
    
    feature_names = ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 
                    'Fasting BS', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina']
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n      === TOP FEATURES ===")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"      {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    return model, scaler, accuracy


def train_all_models():
    """Main function: Auto-detect data, prepare, and train all models"""
    
    print("\n" + "=" * 70)
    print("HEART DISEASE MODEL TRAINING - ALL-IN-ONE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Auto-detect all datasets in training_data/heart_disease/")
    print("  2. Combine them for Generic CVD & CAD models")
    print("  3. Use arrhythmia.data for Arrhythmia model (if available)")
    print("  4. Train all 3 models with optimized settings")
    print("  5. Save models to models/weights/")
    
    X_combined, y_combined, X_arrhythmia, y_arrhythmia = auto_detect_and_load_datasets()
    
    if X_combined is None:
        return False
    
    weights_dir = os.path.join(project_dir, 'models', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    results = {}
    total_start = time.time()
    
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING MODELS")
    print("=" * 70)
    
    print("\n[1/3] Training Generic CVD Model...")
    print("-" * 50)
    model_generic, scaler_generic, acc_generic = train_single_model(
        X_combined, y_combined, 'Generic CVD'
    )
    joblib.dump(model_generic, os.path.join(weights_dir, 'heart_generic_model.pkl'))
    joblib.dump(scaler_generic, os.path.join(weights_dir, 'heart_generic_scaler.pkl'))
    results['Generic CVD'] = acc_generic
    print(f"\n      Saved: heart_generic_model.pkl")
    
    print("\n[2/3] Training CAD (Coronary Artery Disease) Model...")
    print("-" * 50)
    model_cad, scaler_cad, acc_cad = train_single_model(
        X_combined, y_combined, 'CAD'
    )
    joblib.dump(model_cad, os.path.join(weights_dir, 'heart_cad_model.pkl'))
    joblib.dump(scaler_cad, os.path.join(weights_dir, 'heart_cad_scaler.pkl'))
    results['CAD'] = acc_cad
    print(f"\n      Saved: heart_cad_model.pkl")
    
    print("\n[3/3] Training Arrhythmia Model...")
    print("-" * 50)
    model_arrhythmia, scaler_arrhythmia, acc_arrhythmia = train_single_model(
        X_arrhythmia, y_arrhythmia, 'Arrhythmia'
    )
    joblib.dump(model_arrhythmia, os.path.join(weights_dir, 'heart_arrhythmia_model.pkl'))
    joblib.dump(scaler_arrhythmia, os.path.join(weights_dir, 'heart_arrhythmia_scaler.pkl'))
    results['Arrhythmia'] = acc_arrhythmia
    print(f"\n      Saved: heart_arrhythmia_model.pkl")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n=== MODEL ACCURACIES ===")
    for name, acc in results.items():
        print(f"   {name}: {acc:.2%}")
    print(f"\n   Average: {np.mean(list(results.values())):.2%}")
    
    print("\n=== SAVED FILES ===")
    print("   models/weights/heart_generic_model.pkl")
    print("   models/weights/heart_generic_scaler.pkl")
    print("   models/weights/heart_cad_model.pkl")
    print("   models/weights/heart_cad_scaler.pkl")
    print("   models/weights/heart_arrhythmia_model.pkl")
    print("   models/weights/heart_arrhythmia_scaler.pkl")
    
    print("\n=== NEXT STEPS ===")
    print("   1. Restart the Streamlit app")
    print("   2. The app will auto-detect trained models")
    print("   3. Test with real patient data!")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Your heart disease models are ready.")
    print("=" * 70 + "\n")
    
    return True


if __name__ == "__main__":
    train_all_models()

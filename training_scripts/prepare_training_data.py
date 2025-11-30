"""
Prepare Training Data Script
============================
This script prepares the training data for ALL disease models:
- Heart Disease (3 models) - Supports 6 different datasets!
- Pneumonia (3 models) 
- Skin Disease (1 model)

Note: Color Blindness does NOT require training data - it uses interactive clinical tests.

Supported Heart Disease Datasets:
1. cardio_train.csv - Cardiovascular Disease (70,000 records)
2. heart_2022_no_nans.csv - Personal Key Indicators (319,795 records)
3. heart_disease_health_indicators_BRFSS2015.csv - Health Indicators (253,680 records)
4. heart_statlog_cleveland_hungary_final.csv - Combined 5 sources (1,190 records)
5. heart_failure.csv - Heart Failure Prediction (918 records)
6. heart.csv - UCI Original (303 records)

Usage:
------
1. First download the datasets (see COMPREHENSIVE_TRAINING_GUIDE.md)
2. Place datasets in the correct folders under training_data/
3. Run this script: python training_scripts/prepare_training_data.py

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
"""

import pandas as pd
import numpy as np
import os


def load_cardiovascular_dataset(filepath):
    """Load Cardiovascular Disease Dataset (70,000 records)
    Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    """
    print(f"   Loading Cardiovascular Disease Dataset...")
    df = pd.read_csv(filepath, sep=';')
    print(f"   Loaded {len(df):,} records")
    
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
    Source: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
    """
    print(f"   Loading Personal Key Indicators Dataset...")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} records")
    
    age_map = {
        '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
        '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
        '70-74': 72, '75-79': 77, '80 or older': 82
    }
    
    if 'AgeCategory' in df.columns:
        df['age_years'] = df['AgeCategory'].map(age_map).fillna(50)
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
    Source: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
    """
    print(f"   Loading Health Indicators Dataset...")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} records")
    
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
    Source: https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
    """
    print(f"   Loading Comprehensive Dataset (5 sources)...")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} records")
    
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
    Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
    """
    print(f"   Loading Heart Failure Dataset...")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} records")
    
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
    Source: https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv
    """
    print(f"   Loading UCI Original Dataset...")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} records")
    
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        X = df.iloc[:, :9].values.astype(float)
    else:
        X = df[feature_cols].values.astype(float)
    
    target_col = df.columns[-1]
    y = (df[target_col] > 0).astype(int).values
    
    return X, y


def prepare_heart_data(training_data_dir):
    """Prepare heart disease datasets from multiple sources"""
    print("\n" + "=" * 70)
    print("PREPARING HEART DISEASE DATA (Multiple Datasets)")
    print("=" * 70)
    
    heart_dir = os.path.join(training_data_dir, 'heart_disease')
    
    datasets = {
        'cardio_train.csv': ('Cardiovascular Disease (70K)', load_cardiovascular_dataset),
        'heart_2022_no_nans.csv': ('Personal Key Indicators (319K)', load_personal_indicators_dataset),
        'heart_disease_health_indicators_BRFSS2015.csv': ('Health Indicators (253K)', load_health_indicators_dataset),
        'heart_statlog_cleveland_hungary_final.csv': ('Comprehensive (1.1K)', load_comprehensive_dataset),
        'heart_failure.csv': ('Heart Failure (918)', load_heart_failure_dataset),
        'heart.csv': ('UCI Original (303)', load_uci_original_dataset),
    }
    
    X_all = []
    y_all = []
    total_records = 0
    loaded_datasets = 0
    
    print("\n[Step 1] Detecting and loading available datasets...")
    
    for filename, (name, loader) in datasets.items():
        filepath = os.path.join(heart_dir, filename)
        if os.path.exists(filepath):
            try:
                X, y = loader(filepath)
                X_all.append(X)
                y_all.append(y)
                total_records += len(y)
                loaded_datasets += 1
                print(f"   [OK] {name}: {len(y):,} records")
            except Exception as e:
                print(f"   [ERROR] {name}: {str(e)}")
        else:
            print(f"   [SKIP] {name}: File not found")
    
    if loaded_datasets == 0:
        print("\n" + "!" * 70)
        print("ERROR: No heart disease datasets found!")
        print("!" * 70)
        print("\nPlease download at least one dataset:")
        print("1. cardio_train.csv from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        print("2. heart.csv from: https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv")
        print(f"\nPlace files in: {heart_dir}")
        return False
    
    print(f"\n[Step 2] Combining {loaded_datasets} datasets...")
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
    
    print(f"\n   Combined dataset shape: {X_combined.shape}")
    print(f"   Total records: {total_records:,}")
    print(f"   - No disease: {np.sum(y_combined == 0):,} patients ({100*np.sum(y_combined == 0)/len(y_combined):.1f}%)")
    print(f"   - Disease: {np.sum(y_combined == 1):,} patients ({100*np.sum(y_combined == 1)/len(y_combined):.1f}%)")
    
    print("\n[Step 3] Saving prepared datasets...")
    
    np.save(os.path.join(training_data_dir, 'X_generic.npy'), X_combined)
    np.save(os.path.join(training_data_dir, 'y_generic.npy'), y_combined)
    print(f"   Saved: X_generic.npy, y_generic.npy ({total_records:,} records)")
    
    np.save(os.path.join(training_data_dir, 'X_cad.npy'), X_combined)
    np.save(os.path.join(training_data_dir, 'y_cad.npy'), y_combined)
    print(f"   Saved: X_cad.npy, y_cad.npy ({total_records:,} records)")
    
    print("\n[Step 4] Preparing Arrhythmia dataset...")
    arrhythmia_path = os.path.join(training_data_dir, 'arrhythmia', 'arrhythmia.data')
    
    if os.path.exists(arrhythmia_path):
        try:
            arrhythmia_df = pd.read_csv(arrhythmia_path, header=None, na_values='?')
            print(f"   Loaded {len(arrhythmia_df):,} arrhythmia records")
            
            feature_df = arrhythmia_df.iloc[:, :9].copy()
            for col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            
            X_arrhythmia = feature_df.values.astype(float)
            
            for i in range(X_arrhythmia.shape[1]):
                col = X_arrhythmia[:, i]
                mask = np.isnan(col)
                if mask.any():
                    col_mean = np.nanmean(col)
                    if np.isnan(col_mean):
                        col_mean = 0.0
                    col[mask] = col_mean
            
            y_arrhythmia_raw = pd.to_numeric(arrhythmia_df.iloc[:, -1], errors='coerce').fillna(1).values
            y_arrhythmia = (y_arrhythmia_raw > 1).astype(int)
            
            np.save(os.path.join(training_data_dir, 'X_arrhythmia.npy'), X_arrhythmia)
            np.save(os.path.join(training_data_dir, 'y_arrhythmia.npy'), y_arrhythmia)
            print(f"   Saved: X_arrhythmia.npy, y_arrhythmia.npy ({len(y_arrhythmia):,} records)")
            print(f"   - Normal: {np.sum(y_arrhythmia == 0):,} patients")
            print(f"   - Arrhythmia: {np.sum(y_arrhythmia == 1):,} patients")
        except Exception as e:
            print(f"   [ERROR] Arrhythmia: {str(e)}")
            print("   Using combined heart data for arrhythmia model")
            np.save(os.path.join(training_data_dir, 'X_arrhythmia.npy'), X_combined)
            np.save(os.path.join(training_data_dir, 'y_arrhythmia.npy'), y_combined)
    else:
        print("   Arrhythmia data not found - using combined heart data")
        np.save(os.path.join(training_data_dir, 'X_arrhythmia.npy'), X_combined)
        np.save(os.path.join(training_data_dir, 'y_arrhythmia.npy'), y_combined)
        print(f"   Saved: X_arrhythmia.npy, y_arrhythmia.npy ({total_records:,} records)")
    
    print("\n" + "-" * 70)
    print(f"HEART DISEASE DATA PREPARATION COMPLETE!")
    print(f"Total: {total_records:,} records from {loaded_datasets} datasets")
    print("-" * 70)
    
    return True


def check_pneumonia_data(training_data_dir):
    """Check pneumonia dataset structure"""
    print("\n" + "=" * 70)
    print("CHECKING PNEUMONIA DATA")
    print("=" * 70)
    
    pneumonia_dir = os.path.join(training_data_dir, 'pneumonia')
    
    required_structure = {
        'train': ['NORMAL', 'PNEUMONIA'],
        'val': ['NORMAL', 'PNEUMONIA'],
        'test': ['NORMAL', 'PNEUMONIA']
    }
    
    success = True
    total_images = 0
    
    for split, classes in required_structure.items():
        split_path = os.path.join(pneumonia_dir, split)
        
        if not os.path.exists(split_path):
            print(f"   [SKIP] Missing folder: {split_path}")
            success = False
            continue
        
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            
            if not os.path.exists(cls_path):
                print(f"   [SKIP] Missing folder: {cls_path}")
                success = False
                continue
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = [f for f in os.listdir(cls_path) 
                     if os.path.splitext(f)[1].lower() in image_extensions]
            
            total_images += len(images)
            print(f"   [OK] Found {len(images):,} images in {split}/{cls}")
    
    if success:
        print(f"\nPneumonia dataset ready! Total: {total_images:,} images")
    else:
        print(f"\nPneumonia dataset incomplete.")
        print("Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    
    return success


def check_skin_data(training_data_dir):
    """Check skin disease dataset structure"""
    print("\n" + "=" * 70)
    print("CHECKING SKIN DISEASE DATA")
    print("=" * 70)
    
    skin_dir = os.path.join(training_data_dir, 'skin_disease')
    
    metadata_path = os.path.join(skin_dir, 'HAM10000_metadata.csv')
    images_part1 = os.path.join(skin_dir, 'HAM10000_images_part_1')
    images_part2 = os.path.join(skin_dir, 'HAM10000_images_part_2')
    
    success = True
    total_images = 0
    
    if not os.path.exists(metadata_path):
        print(f"   [SKIP] Missing metadata: {metadata_path}")
        success = False
    else:
        metadata_df = pd.read_csv(metadata_path)
        print(f"   [OK] Loaded metadata: {len(metadata_df):,} records")
        
        if 'dx' in metadata_df.columns:
            class_counts = metadata_df['dx'].value_counts()
            print("\n   Class distribution:")
            for cls, count in class_counts.items():
                print(f"      - {cls}: {count:,} images")
    
    for part_name, part_path in [('Part 1', images_part1), ('Part 2', images_part2)]:
        if not os.path.exists(part_path):
            print(f"   [SKIP] Missing folder: {part_path}")
            success = False
        else:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = [f for f in os.listdir(part_path) 
                     if os.path.splitext(f)[1].lower() in image_extensions]
            total_images += len(images)
            print(f"   [OK] Found {len(images):,} images in {part_name}")
    
    if success:
        print(f"\nSkin disease dataset ready! Total: {total_images:,} images")
    else:
        print(f"\nSkin disease dataset incomplete.")
        print("Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    
    return success


def prepare_data():
    print("=" * 70)
    print("PREPARING ALL TRAINING DATA")
    print("=" * 70)
    print("\nNote: Color Blindness does NOT require training data.")
    print("      It uses interactive clinical tests with predefined answers.\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    training_data_dir = os.path.join(project_dir, 'training_data')
    
    results = {
        'heart': False,
        'pneumonia': False,
        'skin': False
    }
    
    results['heart'] = prepare_heart_data(training_data_dir)
    results['pneumonia'] = check_pneumonia_data(training_data_dir)
    results['skin'] = check_skin_data(training_data_dir)
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION SUMMARY")
    print("=" * 70)
    
    status_icons = {True: '[OK]', False: '[X]'}
    
    print(f"\n{status_icons[results['heart']]} Heart Disease Data: {'Ready' if results['heart'] else 'Missing/Incomplete'}")
    print(f"{status_icons[results['pneumonia']]} Pneumonia Data: {'Ready' if results['pneumonia'] else 'Missing/Incomplete'}")
    print(f"{status_icons[results['skin']]} Skin Disease Data: {'Ready' if results['skin'] else 'Missing/Incomplete'}")
    print(f"[i] Color Blindness: No training required (uses interactive tests)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if results['heart']:
        print("\n[OK] Heart Disease: Run 'python training_scripts/train_heart_models.py'")
    else:
        print("\n[X] Heart Disease: Download datasets first (see COMPREHENSIVE_TRAINING_GUIDE.md)")
    
    if results['pneumonia']:
        print("[OK] Pneumonia: Run 'python training_scripts/train_pneumonia_models.py'")
    else:
        print("[X] Pneumonia: Download dataset first (see COMPREHENSIVE_TRAINING_GUIDE.md)")
    
    if results['skin']:
        print("[OK] Skin Disease: Run 'python training_scripts/train_skin_model.py'")
    else:
        print("[X] Skin Disease: Download dataset first (see COMPREHENSIVE_TRAINING_GUIDE.md)")
    
    return all(results.values())


if __name__ == "__main__":
    prepare_data()

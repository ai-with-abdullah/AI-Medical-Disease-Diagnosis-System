"""
Prepare Training Data Script
============================
This script prepares the training data for ALL disease models:
- Heart Disease (3 models)
- Pneumonia (3 models) 
- Skin Disease (1 model)

Note: Color Blindness does NOT require training data - it uses interactive clinical tests.

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

def prepare_heart_data(training_data_dir):
    """Prepare heart disease datasets"""
    print("\n" + "=" * 60)
    print("PREPARING HEART DISEASE DATA")
    print("=" * 60)
    
    success = True
    
    print("\n[Step 1/3] Loading UCI Heart Disease dataset...")
    heart_csv_path = os.path.join(training_data_dir, 'heart_disease', 'heart.csv')
    
    if not os.path.exists(heart_csv_path):
        print(f"WARNING: Could not find {heart_csv_path}")
        print("Please download heart.csv from:")
        print("https://github.com/sharmaroshan/Heart-UCI-Dataset/blob/master/heart.csv")
        print("And place it in: training_data/heart_disease/heart.csv")
        success = False
        X_heart = None
        y_heart = None
    else:
        heart_df = pd.read_csv(heart_csv_path)
        print(f"Loaded {len(heart_df)} records")
        
        feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']
        
        missing_cols = [col for col in feature_cols if col not in heart_df.columns]
        if missing_cols:
            print(f"WARNING: Missing columns: {missing_cols}")
            print(f"Available columns: {list(heart_df.columns)}")
            X_heart = heart_df.iloc[:, :9].values
        else:
            X_heart = heart_df[feature_cols].values
        
        target_col = heart_df.columns[-1]
        y_heart = (heart_df[target_col] > 0).astype(int).values
        
        print(f"Features shape: {X_heart.shape}")
        print(f"   - No disease: {np.sum(y_heart == 0)} patients")
        print(f"   - Disease: {np.sum(y_heart == 1)} patients")
    
    print("\n[Step 2/3] Loading UCI Arrhythmia dataset...")
    arrhythmia_path = os.path.join(training_data_dir, 'arrhythmia', 'arrhythmia.data')
    
    if not os.path.exists(arrhythmia_path):
        print(f"WARNING: Could not find {arrhythmia_path}")
        if X_heart is not None:
            print("Using UCI Heart Disease data for arrhythmia model instead")
            X_arrhythmia = X_heart.copy()
            y_arrhythmia = y_heart.copy()
        else:
            X_arrhythmia = None
            y_arrhythmia = None
    else:
        arrhythmia_df = pd.read_csv(arrhythmia_path, header=None, na_values='?')
        print(f"Loaded {len(arrhythmia_df)} records")
        
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
        
        print(f"Features shape: {X_arrhythmia.shape}")
        print(f"   - Normal: {np.sum(y_arrhythmia == 0)} patients")
        print(f"   - Arrhythmia: {np.sum(y_arrhythmia == 1)} patients")
    
    print("\n[Step 3/3] Saving prepared heart disease datasets...")
    
    if X_heart is not None:
        np.save(os.path.join(training_data_dir, 'X_generic.npy'), X_heart)
        np.save(os.path.join(training_data_dir, 'y_generic.npy'), y_heart)
        print("Saved: X_generic.npy, y_generic.npy")
        
        np.save(os.path.join(training_data_dir, 'X_cad.npy'), X_heart)
        np.save(os.path.join(training_data_dir, 'y_cad.npy'), y_heart)
        print("Saved: X_cad.npy, y_cad.npy")
    
    if X_arrhythmia is not None:
        np.save(os.path.join(training_data_dir, 'X_arrhythmia.npy'), X_arrhythmia)
        np.save(os.path.join(training_data_dir, 'y_arrhythmia.npy'), y_arrhythmia)
        print("Saved: X_arrhythmia.npy, y_arrhythmia.npy")
    
    return success


def check_pneumonia_data(training_data_dir):
    """Check pneumonia dataset structure"""
    print("\n" + "=" * 60)
    print("CHECKING PNEUMONIA DATA")
    print("=" * 60)
    
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
            print(f"WARNING: Missing folder: {split_path}")
            success = False
            continue
        
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            
            if not os.path.exists(cls_path):
                print(f"WARNING: Missing folder: {cls_path}")
                success = False
                continue
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = [f for f in os.listdir(cls_path) 
                     if os.path.splitext(f)[1].lower() in image_extensions]
            
            total_images += len(images)
            print(f"Found {len(images):,} images in {split}/{cls}")
    
    if success:
        print(f"\nPneumonia dataset ready!")
        print(f"Total images: {total_images:,}")
    else:
        print(f"\nPneumonia dataset incomplete.")
        print("Please download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("Extract to: training_data/pneumonia/")
    
    return success


def check_skin_data(training_data_dir):
    """Check skin disease dataset structure"""
    print("\n" + "=" * 60)
    print("CHECKING SKIN DISEASE DATA")
    print("=" * 60)
    
    skin_dir = os.path.join(training_data_dir, 'skin_disease')
    
    metadata_path = os.path.join(skin_dir, 'HAM10000_metadata.csv')
    images_part1 = os.path.join(skin_dir, 'HAM10000_images_part_1')
    images_part2 = os.path.join(skin_dir, 'HAM10000_images_part_2')
    
    success = True
    total_images = 0
    
    if not os.path.exists(metadata_path):
        print(f"WARNING: Missing metadata file: {metadata_path}")
        success = False
    else:
        metadata_df = pd.read_csv(metadata_path)
        print(f"Loaded metadata: {len(metadata_df)} records")
        
        if 'dx' in metadata_df.columns:
            class_counts = metadata_df['dx'].value_counts()
            print("\nClass distribution:")
            for cls, count in class_counts.items():
                print(f"   - {cls}: {count:,} images")
    
    for part_name, part_path in [('Part 1', images_part1), ('Part 2', images_part2)]:
        if not os.path.exists(part_path):
            print(f"WARNING: Missing folder: {part_path}")
            success = False
        else:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = [f for f in os.listdir(part_path) 
                     if os.path.splitext(f)[1].lower() in image_extensions]
            total_images += len(images)
            print(f"Found {len(images):,} images in {part_name}")
    
    if success:
        print(f"\nSkin disease dataset ready!")
        print(f"Total images: {total_images:,}")
    else:
        print(f"\nSkin disease dataset incomplete.")
        print("Please download HAM10000 from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("Extract to: training_data/skin_disease/")
    
    return success


def prepare_data():
    print("=" * 60)
    print("PREPARING ALL TRAINING DATA")
    print("=" * 60)
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
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    
    status_icons = {True: '✅', False: '❌'}
    
    print(f"\n{status_icons[results['heart']]} Heart Disease Data: {'Ready' if results['heart'] else 'Missing/Incomplete'}")
    print(f"{status_icons[results['pneumonia']]} Pneumonia Data: {'Ready' if results['pneumonia'] else 'Missing/Incomplete'}")
    print(f"{status_icons[results['skin']]} Skin Disease Data: {'Ready' if results['skin'] else 'Missing/Incomplete'}")
    print(f"ℹ️  Color Blindness: No training required (uses interactive tests)")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    if results['heart']:
        print("\n✅ Heart Disease: Run 'python training_scripts/train_heart_models.py'")
    else:
        print("\n❌ Heart Disease: Download datasets first (see COMPREHENSIVE_TRAINING_GUIDE.md)")
    
    if results['pneumonia']:
        print("✅ Pneumonia: Run 'python training_scripts/train_pneumonia_models.py'")
    else:
        print("❌ Pneumonia: Download dataset first (see COMPREHENSIVE_TRAINING_GUIDE.md)")
    
    if results['skin']:
        print("✅ Skin Disease: Run 'python training_scripts/train_skin_model.py'")
    else:
        print("❌ Skin Disease: Download dataset first (see COMPREHENSIVE_TRAINING_GUIDE.md)")
    
    return all(results.values())


if __name__ == "__main__":
    prepare_data()

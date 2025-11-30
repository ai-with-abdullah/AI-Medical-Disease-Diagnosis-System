"""
Train Heart Disease Models Script
=================================
This script trains all 3 heart disease models:
1. Generic Cardiovascular Disease (CVD)
2. Coronary Artery Disease (CAD)
3. Cardiac Arrhythmia

Optimized for large datasets (300,000+ records)!

Usage:
------
1. First run: python training_scripts/prepare_training_data.py
2. Then run: python training_scripts/train_heart_models.py

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
"""

import numpy as np
import joblib
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def build_model(n_samples):
    """Build optimized Random Forest model based on dataset size"""
    
    if n_samples > 100000:
        print("   Using LARGE dataset configuration (100K+ records)")
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
        print("   Using MEDIUM dataset configuration (10K+ records)")
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
        print("   Using SMALL dataset configuration (<10K records)")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )


def train_single_model(X, y, disease_type):
    """Train a single model with comprehensive evaluation"""
    
    n_samples = len(y)
    print(f"   Dataset size: {n_samples:,} records")
    print(f"   Class distribution: {np.sum(y==0):,} healthy, {np.sum(y==1):,} disease")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train set: {len(y_train):,}, Test set: {len(y_test):,}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   Building model...")
    model = build_model(n_samples)
    
    print("   Training... (this may take a few minutes for large datasets)")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"   Training completed in {train_time:.1f} seconds")
    
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
    
    print(f"\n   === EVALUATION RESULTS ===")
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1 Score:  {f1:.2%}")
    if auc > 0:
        print(f"   ROC-AUC:   {auc:.2%}")
    
    feature_names = ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 
                    'Fasting BS', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina']
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n   === TOP FEATURES ===")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    return model, scaler, accuracy


def train_all_models():
    print("=" * 70)
    print("TRAINING HEART DISEASE MODELS (Optimized for Large Datasets)")
    print("=" * 70)
    
    training_data_dir = os.path.join(project_dir, 'training_data')
    weights_dir = os.path.join(project_dir, 'models', 'weights')
    
    os.makedirs(weights_dir, exist_ok=True)
    
    results = {}
    total_start = time.time()
    
    print("\n[1/3] Training Generic CVD Model...")
    print("-" * 50)
    X_generic_path = os.path.join(training_data_dir, 'X_generic.npy')
    y_generic_path = os.path.join(training_data_dir, 'y_generic.npy')
    
    if not os.path.exists(X_generic_path):
        print("ERROR: Training data not found!")
        print("Please run prepare_training_data.py first")
        return False
    
    X_generic = np.load(X_generic_path)
    y_generic = np.load(y_generic_path)
    
    model_generic, scaler_generic, acc_generic = train_single_model(
        X_generic, y_generic, 'generic'
    )
    
    joblib.dump(model_generic, os.path.join(weights_dir, 'heart_generic_model.pkl'))
    joblib.dump(scaler_generic, os.path.join(weights_dir, 'heart_generic_scaler.pkl'))
    results['generic'] = acc_generic
    print("\n   Saved: heart_generic_model.pkl, heart_generic_scaler.pkl")
    
    print("\n[2/3] Training CAD (Coronary Artery Disease) Model...")
    print("-" * 50)
    X_cad = np.load(os.path.join(training_data_dir, 'X_cad.npy'))
    y_cad = np.load(os.path.join(training_data_dir, 'y_cad.npy'))
    
    model_cad, scaler_cad, acc_cad = train_single_model(X_cad, y_cad, 'cad')
    
    joblib.dump(model_cad, os.path.join(weights_dir, 'heart_cad_model.pkl'))
    joblib.dump(scaler_cad, os.path.join(weights_dir, 'heart_cad_scaler.pkl'))
    results['cad'] = acc_cad
    print("\n   Saved: heart_cad_model.pkl, heart_cad_scaler.pkl")
    
    print("\n[3/3] Training Arrhythmia Model...")
    print("-" * 50)
    X_arrhythmia = np.load(os.path.join(training_data_dir, 'X_arrhythmia.npy'))
    y_arrhythmia = np.load(os.path.join(training_data_dir, 'y_arrhythmia.npy'))
    
    model_arrhythmia, scaler_arrhythmia, acc_arrhythmia = train_single_model(
        X_arrhythmia, y_arrhythmia, 'arrhythmia'
    )
    
    joblib.dump(model_arrhythmia, os.path.join(weights_dir, 'heart_arrhythmia_model.pkl'))
    joblib.dump(scaler_arrhythmia, os.path.join(weights_dir, 'heart_arrhythmia_scaler.pkl'))
    results['arrhythmia'] = acc_arrhythmia
    print("\n   Saved: heart_arrhythmia_model.pkl, heart_arrhythmia_scaler.pkl")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTotal training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\n=== MODEL ACCURACIES ===")
    print(f"   Generic CVD:  {results['generic']:.2%}")
    print(f"   CAD:          {results['cad']:.2%}")
    print(f"   Arrhythmia:   {results['arrhythmia']:.2%}")
    print(f"\n   Average:      {np.mean(list(results.values())):.2%}")
    
    print("\n=== SAVED FILES ===")
    print("   models/weights/heart_generic_model.pkl")
    print("   models/weights/heart_generic_scaler.pkl")
    print("   models/weights/heart_cad_model.pkl")
    print("   models/weights/heart_cad_scaler.pkl")
    print("   models/weights/heart_arrhythmia_model.pkl")
    print("   models/weights/heart_arrhythmia_scaler.pkl")
    
    print("\n=== NEXT STEPS ===")
    print("   1. Restart the Streamlit app to use trained models")
    print("   2. The app will automatically switch from DEMO to PRODUCTION mode")
    print("   3. Test with real patient data!")
    
    return True


if __name__ == "__main__":
    train_all_models()

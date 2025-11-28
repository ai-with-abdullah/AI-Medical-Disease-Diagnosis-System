"""
Train Heart Disease Models Script
=================================
This script trains all 3 heart disease models:
1. Generic Cardiovascular Disease (CVD)
2. Coronary Artery Disease (CAD)
3. Cardiac Arrhythmia

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

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def build_model():
    """Build Random Forest model for heart disease prediction"""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

def train_single_model(X, y, disease_type):
    """Train a single model and return model + scaler"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    model = build_model()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1 Score:  {f1:.2%}")
    
    return model, scaler, accuracy

def train_all_models():
    print("=" * 60)
    print("TRAINING HEART DISEASE MODELS")
    print("=" * 60)
    
    training_data_dir = os.path.join(project_dir, 'training_data')
    weights_dir = os.path.join(project_dir, 'models', 'weights')
    
    # Ensure weights directory exists
    os.makedirs(weights_dir, exist_ok=True)
    
    results = {}
    
    # ===== GENERIC CVD =====
    print("\n[1/3] Training Generic CVD Model...")
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
    print("   Saved: heart_generic_model.pkl, heart_generic_scaler.pkl")
    
    # ===== CAD =====
    print("\n[2/3] Training CAD (Coronary Artery Disease) Model...")
    X_cad = np.load(os.path.join(training_data_dir, 'X_cad.npy'))
    y_cad = np.load(os.path.join(training_data_dir, 'y_cad.npy'))
    
    model_cad, scaler_cad, acc_cad = train_single_model(X_cad, y_cad, 'cad')
    
    joblib.dump(model_cad, os.path.join(weights_dir, 'heart_cad_model.pkl'))
    joblib.dump(scaler_cad, os.path.join(weights_dir, 'heart_cad_scaler.pkl'))
    results['cad'] = acc_cad
    print("   Saved: heart_cad_model.pkl, heart_cad_scaler.pkl")
    
    # ===== ARRHYTHMIA =====
    print("\n[3/3] Training Arrhythmia Model...")
    X_arrhythmia = np.load(os.path.join(training_data_dir, 'X_arrhythmia.npy'))
    y_arrhythmia = np.load(os.path.join(training_data_dir, 'y_arrhythmia.npy'))
    
    model_arrhythmia, scaler_arrhythmia, acc_arrhythmia = train_single_model(
        X_arrhythmia, y_arrhythmia, 'arrhythmia'
    )
    
    joblib.dump(model_arrhythmia, os.path.join(weights_dir, 'heart_arrhythmia_model.pkl'))
    joblib.dump(scaler_arrhythmia, os.path.join(weights_dir, 'heart_arrhythmia_scaler.pkl'))
    results['arrhythmia'] = acc_arrhythmia
    print("   Saved: heart_arrhythmia_model.pkl, heart_arrhythmia_scaler.pkl")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel Accuracies:")
    print(f"   Generic CVD:  {results['generic']:.2%}")
    print(f"   CAD:          {results['cad']:.2%}")
    print(f"   Arrhythmia:   {results['arrhythmia']:.2%}")
    print("\nTrained models saved to: models/weights/")
    print("\nNext step: Restart the Streamlit app to use trained models")
    
    return True

if __name__ == "__main__":
    train_all_models()

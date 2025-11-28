import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Global variables for each disease type
TRAINED_MODELS = {
    'generic': {'model': None, 'scaler': None, 'loaded': False},
    'cad': {'model': None, 'scaler': None, 'loaded': False},
    'arrhythmia': {'model': None, 'scaler': None, 'loaded': False}
}

def load_trained_models():
    """Load trained models for all disease types if they exist"""
    for disease_type in ['generic', 'cad', 'arrhythmia']:
        if TRAINED_MODELS[disease_type]['loaded']:
            continue
        
        model_path = f'models/weights/heart_{disease_type}_model.pkl'
        scaler_path = f'models/weights/heart_{disease_type}_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                TRAINED_MODELS[disease_type]['model'] = joblib.load(model_path)
                TRAINED_MODELS[disease_type]['scaler'] = joblib.load(scaler_path)
                print(f"✅ Loaded trained {disease_type} model")
            except Exception as e:
                print(f"⚠️ Error loading {disease_type} model: {e}. Using demo mode.")
                TRAINED_MODELS[disease_type]['model'] = None
                TRAINED_MODELS[disease_type]['scaler'] = None
        else:
            print(f"⚠️ {disease_type} model weights not found. Using demo mode.")
            TRAINED_MODELS[disease_type]['model'] = None
            TRAINED_MODELS[disease_type]['scaler'] = None
        
        TRAINED_MODELS[disease_type]['loaded'] = True

def encode_features(features):
    """Encode categorical features to numerical values"""
    sex_map = {'Male': 1, 'Female': 0}
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_map = {'No': 0, 'Yes': 1}
    restecg_map = {'Normal': 0, 'ST-T Abnormality': 1, 'LV Hypertrophy': 2}
    exang_map = {'No': 0, 'Yes': 1}
    
    encoded = {
        'age': features['age'],
        'sex': sex_map.get(features['sex'], 1),
        'cp': cp_map.get(features['cp'], 0),
        'trestbps': features['trestbps'],
        'chol': features['chol'],
        'fbs': fbs_map.get(features['fbs'], 0),
        'restecg': restecg_map.get(features['restecg'], 0),
        'thalach': features['thalach'],
        'exang': exang_map.get(features['exang'], 0)
    }
    
    return encoded

def predict_heart_disease(features, disease_type='generic'):
    """Predict heart disease risk from clinical features
    
    Args:
        features: Dict of clinical parameters
        disease_type: 'generic' (CVD risk), 'cad' (Coronary Artery Disease), 'arrhythmia' (Abnormal heartbeat)
    """
    
    # Load trained models
    load_trained_models()
    
    model = TRAINED_MODELS[disease_type]['model']
    scaler = TRAINED_MODELS[disease_type]['scaler']
    
    # Encode features
    encoded_features = encode_features(features)
    
    # Create feature array
    feature_array = np.array([
        encoded_features['age'],
        encoded_features['sex'],
        encoded_features['cp'],
        encoded_features['trestbps'],
        encoded_features['chol'],
        encoded_features['fbs'],
        encoded_features['restecg'],
        encoded_features['thalach'],
        encoded_features['exang']
    ]).reshape(1, -1)
    
    # Disease type descriptions
    disease_labels = {
        'generic': 'General Cardiovascular Disease',
        'cad': 'Coronary Artery Disease (CAD)',
        'arrhythmia': 'Cardiac Arrhythmia (Abnormal Heartbeat)'
    }
    
    disease_descriptions = {
        'generic': 'Generic heart disease risk (Yes/No prediction)',
        'cad': 'Coronary Artery Disease - blockage in heart arteries',
        'arrhythmia': 'Arrhythmia - irregular heartbeat patterns'
    }
    
    if model is not None and scaler is not None:
        # PRODUCTION MODE: Use trained Random Forest model
        feature_scaled = scaler.transform(feature_array)
        
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0][1]  # Probability of disease
        
        # Risk level classification
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Get feature importances from trained model
        feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Blood Pressure', 'Cholesterol', 
                        'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_data = {
                'feature': feature_names,
                'importance': [float(imp) for imp in importances]
            }
        else:
            # Default importance if not available
            feature_importance_data = {
                'feature': feature_names,
                'importance': [0.25, 0.03, 0.06, 0.20, 0.18, 0.02, 0.01, 0.15, 0.10]
            }
        
        return {
            'risk_level': risk_level,
            'probability': float(probability),
            'model': f'Random Forest Classifier (Trained - {disease_type.upper()})',
            'feature_importance': feature_importance_data,
            'disease_type': disease_type,
            'disease_label': disease_labels[disease_type],
            'disease_description': disease_descriptions[disease_type]
        }
    
    else:
        # DEMO MODE: Use risk score calculation
        age_score = min(encoded_features['age'] / 100.0, 1.0)
        bp_score = max(0, (encoded_features['trestbps'] - 120) / 80.0)
        chol_score = max(0, (encoded_features['chol'] - 200) / 400.0)
        hr_score = 1.0 - (encoded_features['thalach'] / 220.0)
        
        risk_score = (age_score * 0.3 + bp_score * 0.25 + chol_score * 0.25 + 
                      hr_score * 0.2 + encoded_features['exang'] * 0.15 + 
                      encoded_features['cp'] * 0.05)
        
        probability = np.clip(risk_score, 0.1, 0.95)
        
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        feature_importance_data = {
            'feature': ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate', 
                        'Exercise Angina', 'Chest Pain Type', 'Sex', 'Fasting Blood Sugar', 'Resting ECG'],
            'importance': [0.25, 0.20, 0.18, 0.15, 0.10, 0.06, 0.03, 0.02, 0.01]
        }
        
        return {
            'risk_level': risk_level,
            'probability': float(probability),
            'model': f'Random Forest Classifier (Demo - {disease_type.upper()})',
            'feature_importance': feature_importance_data,
            'disease_type': disease_type,
            'disease_label': disease_labels[disease_type],
            'disease_description': disease_descriptions[disease_type]
        }

def build_heart_disease_model():
    """Build Random Forest model for heart disease prediction"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    return model

def train_heart_model(X_train, y_train, disease_type='generic'):
    """Train heart disease prediction model for specific disease type
    
    Args:
        X_train: Training features
        y_train: Training labels
        disease_type: 'generic', 'cad', or 'arrhythmia'
    """
    model = build_heart_disease_model()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def is_using_trained_model(disease_type='generic'):
    """Check if system is using trained model or demo mode for specific disease type"""
    load_trained_models()
    return TRAINED_MODELS[disease_type]['model'] is not None and TRAINED_MODELS[disease_type]['scaler'] is not None

def get_disease_types():
    """Get list of available disease types"""
    return {
        'generic': {
            'label': 'Generic Cardiovascular Disease',
            'description': 'General heart disease risk assessment (Yes/No)',
            'icon': '❤️'
        },
        'cad': {
            'label': 'Coronary Artery Disease (CAD)',
            'description': 'Blockage in heart arteries - specific CAD detection',
            'icon': '💔'
        },
        'arrhythmia': {
            'label': 'Cardiac Arrhythmia (Abnormal Heartbeat)',
            'description': 'Irregular heartbeat detection and classification',
            'icon': '📈'
        }
    }

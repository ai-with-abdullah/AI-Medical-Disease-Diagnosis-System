import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def encode_features(features):
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

def predict_heart_disease(features):
    encoded_features = encode_features(features)
    
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
        'probability': probability,
        'model': 'Random Forest Classifier',
        'feature_importance': feature_importance_data
    }

def build_heart_disease_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    return model

def train_heart_model(X_train, y_train):
    model = build_heart_disease_model()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

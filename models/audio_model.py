import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os
import pickle
from PIL import Image

TRAINED_AUDIO_MODELS_LOADED = False
TRAINED_AUDIO_MODELS = {}

def load_trained_audio_models():
    """Load trained audio models if they exist, otherwise use rule-based mode"""
    global TRAINED_AUDIO_MODELS_LOADED, TRAINED_AUDIO_MODELS
    
    if TRAINED_AUDIO_MODELS_LOADED:
        return TRAINED_AUDIO_MODELS
    
    model_paths = {
        'RandomForest': {
            'model': 'models/weights/pneumonia_audio_rf_model.pkl',
            'scaler': 'models/weights/pneumonia_audio_rf_scaler.pkl'
        },
        'NeuralNetwork': {
            'model': 'models/weights/pneumonia_audio_nn_model.h5',
            'scaler': 'models/weights/pneumonia_audio_nn_scaler.pkl'
        }
    }
    
    for model_name, paths in model_paths.items():
        model_path = paths['model']
        scaler_path = paths['scaler']
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                if model_name == 'RandomForest':
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    try:
                        from tensorflow import keras
                        model = keras.models.load_model(model_path)
                    except ImportError:
                        print(f"⚠️ TensorFlow not available for {model_name}")
                        continue
                
                TRAINED_AUDIO_MODELS[model_name] = {
                    'model': model,
                    'scaler': scaler
                }
                print(f"✅ Loaded trained {model_name} audio model from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading {model_name}: {e}. Using demo mode.")
                TRAINED_AUDIO_MODELS[model_name] = None
        else:
            print(f"⚠️ {model_name} audio weights not found. Using demo mode.")
            TRAINED_AUDIO_MODELS[model_name] = None
    
    TRAINED_AUDIO_MODELS_LOADED = True
    return TRAINED_AUDIO_MODELS


def extract_features_for_prediction(audio_file, sr=22050, duration=10):
    """
    Extract comprehensive audio features for model prediction.
    Same feature extraction as training script (97 features).
    """
    try:
        y, sr = librosa.load(audio_file, sr=sr, duration=duration)
        
        if len(y) < sr * 0.5:
            return None
        
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)), mode='constant')
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        rms = np.mean(librosa.feature.rms(y=y))
        
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid],
            [spectral_rolloff],
            [spectral_bandwidth],
            [zcr],
            [rms],
            chroma
        ])
        
        return features
        
    except Exception as e:
        print(f"   Warning: Could not extract features: {e}")
        return None


def predict_with_trained_model(audio_file, precomputed_features=None):
    """
    Use trained model to predict pneumonia from audio.
    
    Args:
        audio_file: Path to audio file
        precomputed_features: Pre-extracted features (97-dim) to avoid double extraction
    """
    load_trained_audio_models()
    
    if 'RandomForest' in TRAINED_AUDIO_MODELS and TRAINED_AUDIO_MODELS['RandomForest'] is not None:
        model_info = TRAINED_AUDIO_MODELS['RandomForest']
        model = model_info['model']
        scaler = model_info['scaler']
        
        if precomputed_features is not None:
            features = precomputed_features
        else:
            features = extract_features_for_prediction(audio_file)
        
        if features is None:
            return None, None, 'RandomForest'
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        prediction_proba = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        
        if prediction == 1:
            result = 'Abnormal - Possible Pneumonia'
            confidence = float(prediction_proba[1])
        else:
            result = 'Normal'
            confidence = float(prediction_proba[0])
        
        return result, confidence, 'RandomForest (Trained)'
    
    elif 'NeuralNetwork' in TRAINED_AUDIO_MODELS and TRAINED_AUDIO_MODELS['NeuralNetwork'] is not None:
        model_info = TRAINED_AUDIO_MODELS['NeuralNetwork']
        model = model_info['model']
        scaler = model_info['scaler']
        
        if precomputed_features is not None:
            features = precomputed_features
        else:
            features = extract_features_for_prediction(audio_file)
        
        if features is None:
            return None, None, 'NeuralNetwork'
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        prediction_proba = model.predict(features_scaled, verbose=0)[0]
        prediction = np.argmax(prediction_proba)
        
        if prediction == 1:
            result = 'Abnormal - Possible Pneumonia'
            confidence = float(prediction_proba[1])
        else:
            result = 'Normal'
            confidence = float(prediction_proba[0])
        
        return result, confidence, 'NeuralNetwork (Trained)'
    
    return None, None, None


def is_using_trained_audio_models():
    """Check if system is using trained audio models or demo mode"""
    load_trained_audio_models()
    trained_count = sum(1 for model in TRAINED_AUDIO_MODELS.values() if model is not None)
    return trained_count > 0, trained_count, len(TRAINED_AUDIO_MODELS)


def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050, duration=10)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    features = {
        'mfcc_mean': mfcc_mean,
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'zero_crossing_rate': np.mean(zero_crossing_rate),
        'chroma_mean': np.mean(chroma, axis=1),
        'raw_audio': y,
        'sr': sr,
        'mfcc_full': mfcc
    }
    
    return features

def generate_mfcc_plot(mfcc, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
    ax.set_title('MFCC Features')
    ax.set_ylabel('MFCC Coefficients')
    fig.colorbar(img, ax=ax)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def generate_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, x_axis='time', y_axis='hz', sr=sr, ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def classify_audio_type(features):
    zcr = features['zero_crossing_rate']
    spectral_centroid = features['spectral_centroid']
    
    if zcr > 0.1 and spectral_centroid > 2000:
        return 'Cough'
    else:
        return 'Breathing'

def analyze_audio(audio_file):
    """
    Analyze audio file for pneumonia detection.
    Uses trained models if available, otherwise falls back to rule-based analysis.
    
    Returns error result if audio file cannot be processed.
    """
    try:
        features = extract_audio_features(audio_file)
    except Exception as e:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'audio_type': 'Unknown',
            'mfcc_plot': None,
            'spectrogram': None,
            'model_used': 'N/A',
            'using_trained_model': False,
            'error': f'Could not process audio file: {str(e)}',
            'features': {}
        }
    
    audio_type = classify_audio_type(features)
    
    ml_features = extract_features_for_prediction(audio_file)
    trained_prediction, trained_confidence, model_used = predict_with_trained_model(
        audio_file, precomputed_features=ml_features
    )
    
    if trained_prediction is not None and trained_confidence is not None:
        prediction = trained_prediction
        confidence = trained_confidence
        using_trained_model = True
    else:
        mfcc_score = np.mean(np.abs(features['mfcc_mean']))
        spectral_score = features['spectral_centroid'] / 5000.0
        zcr_score = features['zero_crossing_rate'] * 2
        
        combined_score = (mfcc_score * 0.5 + spectral_score * 0.3 + zcr_score * 0.2)
        
        if combined_score > 0.6:
            prediction = 'Abnormal - Possible Pneumonia'
            confidence = 0.65 + (combined_score - 0.6) * 0.7
        else:
            prediction = 'Normal'
            confidence = 0.65 + (0.6 - combined_score) * 0.7
        
        confidence = np.clip(confidence, 0.60, 0.95)
        model_used = 'Rule-based (Demo Mode)'
        using_trained_model = False
    
    try:
        mfcc_plot = generate_mfcc_plot(features['mfcc_full'], features['sr'])
        spectrogram = generate_spectrogram(features['raw_audio'], features['sr'])
    except Exception as e:
        print(f"Warning: Could not generate audio plots: {e}")
        mfcc_plot = None
        spectrogram = None
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'audio_type': audio_type,
        'mfcc_plot': mfcc_plot,
        'spectrogram': spectrogram,
        'model_used': model_used,
        'using_trained_model': using_trained_model,
        'features': {
            'spectral_centroid': features['spectral_centroid'],
            'zero_crossing_rate': features['zero_crossing_rate'],
            'mfcc_mean_value': np.mean(features['mfcc_mean'])
        }
    }

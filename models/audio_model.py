import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image

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
    features = extract_audio_features(audio_file)
    
    audio_type = classify_audio_type(features)
    
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
    
    mfcc_plot = generate_mfcc_plot(features['mfcc_full'], features['sr'])
    spectrogram = generate_spectrogram(features['raw_audio'], features['sr'])
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'audio_type': audio_type,
        'mfcc_plot': mfcc_plot,
        'spectrogram': spectrogram,
        'features': {
            'spectral_centroid': features['spectral_centroid'],
            'zero_crossing_rate': features['zero_crossing_rate'],
            'mfcc_mean_value': np.mean(features['mfcc_mean'])
        }
    }

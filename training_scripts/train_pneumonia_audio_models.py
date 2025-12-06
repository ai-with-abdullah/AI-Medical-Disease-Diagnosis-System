"""
Train Pneumonia Audio Detection Models
======================================
This script trains audio classification models for pneumonia detection
from cough and breathing sounds.

Models trained:
1. Random Forest - Fast, reliable baseline
2. Neural Network (MLP) - Higher accuracy

Supported Datasets (6 total, auto-detected):
1. COUGHVID (25,000+ recordings) - folder: coughvid/
2. Coswara (2,635 individuals) - folder: coswara/
3. ICBHI 2017 (920 recordings) - folder: icbhi_2017/
4. Virufy COVID-19 (1,000+ recordings) - folder: virufy/
5. COVID-19 Cough Audio (4,000+ recordings) - folder: covid_cough/
6. Kaggle Respiratory Sound (5,500 recordings) - folder: kaggle_respiratory/

Usage:
------
1. Download datasets and place in training_data/pneumonia_audio/ folder
2. Run: python training_scripts/train_pneumonia_audio_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle
import json
import time
import signal
import hashlib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DATA_DIR = str(PROJECT_ROOT / 'training_data' / 'pneumonia_audio')
WEIGHTS_DIR = str(PROJECT_ROOT / 'models' / 'weights')
CACHE_DIR = str(PROJECT_ROOT / 'training_data' / '.feature_cache')

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa not installed. Run: pip install librosa")
    LIBROSA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not installed. Run: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not installed. Neural Network training will be skipped.")
    TF_AVAILABLE = False


def timeout_handler(signum, frame):
    """Handler for timeout signal"""
    raise TimeoutError("Audio processing timed out")


def extract_audio_features_single(audio_path, sr=22050, duration=8, timeout_seconds=30):
    """
    Extract comprehensive audio features from an audio file with timeout.
    
    Features extracted (97 total):
    - MFCC mean (40)
    - MFCC std (40)
    - Spectral centroid mean (1)
    - Spectral rolloff mean (1)
    - Spectral bandwidth mean (1)
    - Zero crossing rate mean (1)
    - RMS energy mean (1)
    - Chroma mean (12)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        if len(y) < sr * 0.3:
            return None
        
        min_length = sr * 2
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)), mode='constant')
        
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
        return None


def process_single_file(args):
    """Process a single audio file - for use with multiprocessing Pool"""
    idx, file_path, label, cache_path = args
    
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                return idx, cached['features'], label, True
        except:
            pass
    
    features = extract_audio_features_single(file_path)
    
    if features is not None and cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'features': features}, f)
        except:
            pass
    
    return idx, features, label, False


def get_cache_path(file_path):
    """Generate cache path for a given audio file"""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}.pkl")


def detect_datasets():
    """Detect which audio datasets are available in training_data/pneumonia_audio/"""
    datasets = {}
    
    if not os.path.exists(AUDIO_DATA_DIR):
        print(f"⚠️ Audio data directory not found: {AUDIO_DATA_DIR}")
        print("   Creating directory...")
        os.makedirs(AUDIO_DATA_DIR, exist_ok=True)
        return datasets
    
    coughvid_dir = os.path.join(AUDIO_DATA_DIR, 'coughvid')
    if os.path.exists(coughvid_dir):
        audio_files = []
        for root, dirs, files in os.walk(coughvid_dir):
            for f in files:
                if f.endswith(('.wav', '.webm', '.ogg', '.mp3')):
                    audio_files.append(os.path.join(root, f))
        if audio_files:
            datasets['coughvid'] = {
                'path': coughvid_dir,
                'files': audio_files,
                'count': len(audio_files)
            }
            print(f"✅ Found COUGHVID: {len(audio_files)} audio files")
    
    coswara_dir = os.path.join(AUDIO_DATA_DIR, 'coswara')
    if os.path.exists(coswara_dir):
        audio_files = []
        for root, dirs, files in os.walk(coswara_dir):
            for f in files:
                if f.endswith(('.wav', '.mp3')) and ('cough' in f.lower() or 'breathing' in f.lower()):
                    audio_files.append(os.path.join(root, f))
        if audio_files:
            datasets['coswara'] = {
                'path': coswara_dir,
                'files': audio_files,
                'count': len(audio_files)
            }
            print(f"✅ Found Coswara: {len(audio_files)} audio files")
    
    icbhi_dir = os.path.join(AUDIO_DATA_DIR, 'icbhi_2017')
    if os.path.exists(icbhi_dir):
        audio_files = []
        for root, dirs, files in os.walk(icbhi_dir):
            for f in files:
                if f.endswith('.wav'):
                    audio_files.append(os.path.join(root, f))
        if audio_files:
            datasets['icbhi_2017'] = {
                'path': icbhi_dir,
                'files': audio_files,
                'count': len(audio_files)
            }
            print(f"✅ Found ICBHI 2017: {len(audio_files)} audio files")
    
    virufy_dir = os.path.join(AUDIO_DATA_DIR, 'virufy')
    if os.path.exists(virufy_dir):
        audio_files = []
        for root, dirs, files in os.walk(virufy_dir):
            for f in files:
                if f.endswith(('.wav', '.mp3', '.ogg')):
                    audio_files.append(os.path.join(root, f))
        if audio_files:
            datasets['virufy'] = {
                'path': virufy_dir,
                'files': audio_files,
                'count': len(audio_files)
            }
            print(f"✅ Found Virufy: {len(audio_files)} audio files")
    
    covid_cough_dir = os.path.join(AUDIO_DATA_DIR, 'covid_cough')
    if os.path.exists(covid_cough_dir):
        audio_files = []
        for root, dirs, files in os.walk(covid_cough_dir):
            for f in files:
                if f.endswith(('.wav', '.mp3', '.ogg')):
                    audio_files.append(os.path.join(root, f))
        if audio_files:
            datasets['covid_cough'] = {
                'path': covid_cough_dir,
                'files': audio_files,
                'count': len(audio_files)
            }
            print(f"✅ Found COVID Cough: {len(audio_files)} audio files")
    
    kaggle_resp_dir = os.path.join(AUDIO_DATA_DIR, 'kaggle_respiratory')
    if os.path.exists(kaggle_resp_dir):
        audio_files = []
        for root, dirs, files in os.walk(kaggle_resp_dir):
            for f in files:
                if f.endswith('.wav'):
                    audio_files.append(os.path.join(root, f))
        if audio_files:
            datasets['kaggle_respiratory'] = {
                'path': kaggle_resp_dir,
                'files': audio_files,
                'count': len(audio_files)
            }
            print(f"✅ Found Kaggle Respiratory: {len(audio_files)} audio files")
    
    organized_dir = os.path.join(AUDIO_DATA_DIR, 'organized')
    if os.path.exists(organized_dir):
        normal_dir = os.path.join(organized_dir, 'normal')
        abnormal_dir = os.path.join(organized_dir, 'abnormal')
        
        if os.path.exists(normal_dir) and os.path.exists(abnormal_dir):
            normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                          if f.endswith(('.wav', '.mp3', '.ogg'))]
            abnormal_files = [os.path.join(abnormal_dir, f) for f in os.listdir(abnormal_dir)
                            if f.endswith(('.wav', '.mp3', '.ogg'))]
            
            if normal_files or abnormal_files:
                datasets['organized'] = {
                    'path': organized_dir,
                    'normal': normal_files,
                    'abnormal': abnormal_files,
                    'count': len(normal_files) + len(abnormal_files)
                }
                print(f"✅ Found Organized data: {len(normal_files)} normal, {len(abnormal_files)} abnormal")
    
    return datasets


def load_coughvid_data(dataset_info):
    """Load COUGHVID dataset with labels from metadata"""
    print("\n📂 Loading COUGHVID dataset...")
    
    coughvid_dir = dataset_info['path']
    metadata_path = os.path.join(coughvid_dir, 'metadata_compiled.csv')
    
    audio_files = dataset_info['files']
    labels = []
    valid_files = []
    
    if os.path.exists(metadata_path):
        try:
            metadata = pd.read_csv(metadata_path)
            
            label_col = None
            for col in ['status', 'covid_status', 'label', 'diagnosis']:
                if col in metadata.columns:
                    label_col = col
                    break
            
            if label_col:
                file_to_label = {}
                id_col = 'uuid' if 'uuid' in metadata.columns else metadata.columns[0]
                
                for _, row in metadata.iterrows():
                    file_id = str(row[id_col])
                    status = str(row[label_col]).lower()
                    
                    if 'healthy' in status or 'negative' in status or status == '0':
                        file_to_label[file_id] = 0
                    elif 'covid' in status or 'positive' in status or 'symptomatic' in status or status == '1':
                        file_to_label[file_id] = 1
                
                for f in audio_files:
                    file_id = Path(f).stem
                    if file_id in file_to_label:
                        valid_files.append(f)
                        labels.append(file_to_label[file_id])
                
                print(f"   Loaded {len(valid_files)} labeled files from metadata")
                return valid_files, labels
        except Exception as e:
            print(f"   Warning: Could not read metadata: {e}")
    
    print("   No metadata found, using folder-based labels...")
    for f in audio_files:
        path_lower = f.lower()
        if 'healthy' in path_lower or 'normal' in path_lower or 'negative' in path_lower:
            valid_files.append(f)
            labels.append(0)
        elif 'covid' in path_lower or 'positive' in path_lower or 'symptomatic' in path_lower or 'abnormal' in path_lower:
            valid_files.append(f)
            labels.append(1)
    
    print(f"   Loaded {len(valid_files)} files with folder-based labels")
    return valid_files, labels


def load_organized_data(dataset_info):
    """Load pre-organized normal/abnormal audio files"""
    print("\n📂 Loading organized dataset...")
    
    normal_files = dataset_info.get('normal', [])
    abnormal_files = dataset_info.get('abnormal', [])
    
    audio_files = normal_files + abnormal_files
    labels = [0] * len(normal_files) + [1] * len(abnormal_files)
    
    print(f"   Loaded {len(normal_files)} normal, {len(abnormal_files)} abnormal files")
    return audio_files, labels


def load_folder_based_data(dataset_info, dataset_name):
    """Load dataset based on folder structure (normal/abnormal, healthy/covid, etc.)"""
    print(f"\n📂 Loading {dataset_name} dataset...")
    
    base_path = dataset_info['path']
    audio_files = []
    labels = []
    
    normal_patterns = ['normal', 'healthy', 'negative', 'neg']
    abnormal_patterns = ['abnormal', 'covid', 'positive', 'pos', 'pneumonia', 'crackle', 'wheeze']
    
    for root, dirs, files in os.walk(base_path):
        folder_name = os.path.basename(root).lower()
        parent_folder = os.path.basename(os.path.dirname(root)).lower()
        
        is_normal = any(p in folder_name for p in normal_patterns) or any(p in parent_folder for p in normal_patterns)
        is_abnormal = any(p in folder_name for p in abnormal_patterns) or any(p in parent_folder for p in abnormal_patterns)
        
        if is_normal or is_abnormal:
            for f in files:
                if f.endswith(('.wav', '.mp3', '.ogg', '.webm')):
                    audio_files.append(os.path.join(root, f))
                    labels.append(0 if is_normal else 1)
    
    print(f"   Loaded {len(audio_files)} files ({labels.count(0)} normal, {labels.count(1)} abnormal)")
    return audio_files, labels


def combine_datasets(datasets):
    """Combine all detected datasets into unified training data"""
    print("\n" + "=" * 60)
    print("COMBINING DATASETS")
    print("=" * 60)
    
    all_files = []
    all_labels = []
    
    for name, info in datasets.items():
        if name == 'coughvid':
            files, labels = load_coughvid_data(info)
        elif name == 'organized':
            files, labels = load_organized_data(info)
        else:
            files, labels = load_folder_based_data(info, name)
        
        all_files.extend(files)
        all_labels.extend(labels)
    
    print(f"\n📊 Combined dataset: {len(all_files)} total audio files")
    print(f"   Normal: {all_labels.count(0)}, Abnormal: {all_labels.count(1)}")
    
    return all_files, all_labels


def extract_features_parallel(audio_files, labels, max_workers=None, sample_size=None, use_cache=True):
    """Extract features from audio files using multiprocessing Pool
    
    Args:
        audio_files: List of audio file paths
        labels: List of corresponding labels
        max_workers: Number of parallel workers (default: CPU count - 1)
        sample_size: If set, randomly sample this many files for faster training
        use_cache: Whether to cache extracted features for faster reprocessing
    """
    print("\n" + "=" * 60)
    print("EXTRACTING AUDIO FEATURES")
    print("=" * 60)
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    if sample_size and sample_size < len(audio_files):
        print(f"⚡ FAST MODE: Sampling {sample_size} files from {len(audio_files)} total")
        np.random.seed(42)
        indices = np.random.choice(len(audio_files), sample_size, replace=False)
        audio_files = [audio_files[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    total_files = len(audio_files)
    print(f"Processing {total_files} audio files...")
    print(f"   Using {max_workers} parallel processes (multiprocessing)")
    print(f"   Feature caching: {'ENABLED' if use_cache else 'DISABLED'}")
    print(f"   Estimated time: ~{max(1, (total_files * 2) // (60 * max_workers))} - {max(2, (total_files * 4) // (60 * max_workers))} minutes")
    print("")
    
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    tasks = []
    for idx, (file_path, label) in enumerate(zip(audio_files, labels)):
        cache_path = get_cache_path(file_path) if use_cache else None
        tasks.append((idx, file_path, label, cache_path))
    
    start_time = time.time()
    features_list = []
    valid_labels = []
    cached_count = 0
    processed_count = 0
    failed_count = 0
    
    try:
        with Pool(processes=max_workers) as pool:
            results = []
            
            for result in pool.imap_unordered(process_single_file, tasks, chunksize=max(1, total_files // (max_workers * 10))):
                processed_count += 1
                idx, features, label, from_cache = result
                
                if features is not None:
                    features_list.append(features)
                    valid_labels.append(label)
                    if from_cache:
                        cached_count += 1
                else:
                    failed_count += 1
                
                if processed_count == 1 or processed_count % 50 == 0 or processed_count == total_files:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_files - processed_count) / rate if rate > 0 else 0
                    progress_pct = (processed_count / total_files) * 100
                    
                    status_parts = [f"[{progress_pct:5.1f}%]", f"{processed_count}/{total_files}"]
                    status_parts.append(f"Speed: {rate:.1f}/s")
                    status_parts.append(f"ETA: {remaining/60:.1f}m")
                    if cached_count > 0:
                        status_parts.append(f"(cached: {cached_count})")
                    
                    print("   " + " | ".join(status_parts))
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user!")
        print(f"   Processed {processed_count} files before interrupt")
        if len(features_list) >= 100:
            print(f"   Continuing with {len(features_list)} successfully processed files...")
        else:
            print("   Not enough data to continue. Exiting.")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n⚠️ Error during parallel processing: {e}")
        print("   Attempting to continue with processed files...")
    
    if len(features_list) == 0:
        print("\n❌ No features extracted! Check your audio files.")
        return np.array([]), np.array([])
    
    X = np.array(features_list)
    y = np.array(valid_labels)
    
    total_time = time.time() - start_time
    print(f"\n✅ Feature extraction complete in {total_time/60:.1f} minutes!")
    print(f"   Successfully processed: {len(X)} files")
    print(f"   Failed/skipped: {failed_count} files")
    print(f"   Loaded from cache: {cached_count} files")
    print(f"   Feature vector size: {X.shape[1] if len(X) > 0 else 0}")
    print(f"   Class distribution: Normal={sum(y==0)}, Abnormal={sum(y==1)}")
    
    return X, y


def train_random_forest(X_train, X_test, y_train, y_test, class_weights):
    """Train Random Forest classifier"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Random Forest Training Complete!")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    cv_scores = cross_val_score(rf_model, np.vstack([X_train, X_test]), 
                                np.concatenate([y_train, y_test]), cv=5)
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return rf_model, accuracy


def train_neural_network(X_train, X_test, y_train, y_test, class_weights):
    """Train Neural Network (MLP) classifier"""
    if not TF_AVAILABLE:
        print("\n⚠️ TensorFlow not available. Skipping Neural Network training.")
        return None, 0
    
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK MODEL")
    print("=" * 60)
    
    input_dim = X_train.shape[1]
    
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    print("\nTraining Neural Network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Neural Network Training Complete!")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    return model, accuracy


def save_models(rf_model, rf_scaler, nn_model, nn_scaler):
    """Save trained models to disk"""
    print("\n" + "=" * 60)
    print("SAVING TRAINED MODELS")
    print("=" * 60)
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    rf_model_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_rf_model.pkl')
    rf_scaler_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_rf_scaler.pkl')
    
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✅ Saved Random Forest model: {rf_model_path}")
    
    with open(rf_scaler_path, 'wb') as f:
        pickle.dump(rf_scaler, f)
    print(f"✅ Saved RF scaler: {rf_scaler_path}")
    
    if nn_model is not None and TF_AVAILABLE:
        nn_model_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_nn_model.h5')
        nn_scaler_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_nn_scaler.pkl')
        
        nn_model.save(nn_model_path)
        print(f"✅ Saved Neural Network model: {nn_model_path}")
        
        with open(nn_scaler_path, 'wb') as f:
            pickle.dump(nn_scaler, f)
        print(f"✅ Saved NN scaler: {nn_scaler_path}")


def train_audio_models(fast_mode=False, sample_size=2000, workers=None, no_cache=False):
    """Main training function
    
    Args:
        fast_mode: If True, use only a sample of the data for faster training
        sample_size: Number of files to use in fast mode (default 2000)
        workers: Number of parallel workers for feature extraction (default: CPU count - 1)
        no_cache: If True, disable feature caching
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)
    
    print("\n" + "=" * 60)
    print("PNEUMONIA AUDIO MODEL TRAINING")
    print("=" * 60)
    print("Training audio classification models for pneumonia detection")
    print("Models: Random Forest + Neural Network (MLP)")
    if fast_mode:
        print(f"⚡ FAST MODE ENABLED - Using {sample_size} samples")
    print(f"   Using {workers} CPU cores for parallel processing")
    print("=" * 60)
    
    if not LIBROSA_AVAILABLE or not SKLEARN_AVAILABLE:
        print("\n❌ Missing required libraries. Please install:")
        if not LIBROSA_AVAILABLE:
            print("   pip install librosa")
        if not SKLEARN_AVAILABLE:
            print("   pip install scikit-learn")
        return False
    
    print("\n📁 Scanning for datasets in:", AUDIO_DATA_DIR)
    datasets = detect_datasets()
    
    if not datasets:
        print("\n" + "=" * 60)
        print("NO DATASETS FOUND!")
        print("=" * 60)
        print(f"\nPlease download audio datasets and place them in:")
        print(f"   {AUDIO_DATA_DIR}/")
        print("\nSupported dataset folders:")
        print("   - coughvid/          (COUGHVID dataset)")
        print("   - coswara/           (Coswara dataset)")
        print("   - icbhi_2017/        (ICBHI 2017 Respiratory)")
        print("   - virufy/            (Virufy COVID-19)")
        print("   - covid_cough/       (COVID Cough Audio)")
        print("   - kaggle_respiratory/ (Kaggle Respiratory Sound)")
        print("   - organized/         (Pre-organized normal/abnormal folders)")
        print("\nSee COMPREHENSIVE_TRAINING_GUIDE.md for download links!")
        return False
    
    audio_files, labels = combine_datasets(datasets)
    
    if len(audio_files) < 100:
        print(f"\n⚠️ Only {len(audio_files)} labeled audio files found.")
        print("   Recommended: At least 1,000 files for reliable training.")
        print("   Download more datasets for better accuracy!")
        
        if len(audio_files) < 20:
            print("\n❌ Not enough data for training. Need at least 20 labeled files.")
            return False
    
    X, y = extract_features_parallel(
        audio_files, labels, 
        max_workers=workers,
        sample_size=sample_size if fast_mode else None,
        use_cache=not no_cache
    )
    
    if len(X) < 20:
        print("\n❌ Not enough valid audio files for training.")
        return False
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"   Class weights: {class_weights}")
    
    rf_model, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test, class_weights)
    
    nn_model, nn_accuracy = train_neural_network(X_train, X_test, y_train, y_test, class_weights)
    
    save_models(rf_model, scaler, nn_model, scaler)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel Performance Summary:")
    print(f"   Random Forest Accuracy: {rf_accuracy*100:.2f}%")
    if nn_model is not None:
        print(f"   Neural Network Accuracy: {nn_accuracy*100:.2f}%")
    print("\nModel files saved in:", WEIGHTS_DIR)
    print("\nRestart the app to use trained models!")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Pneumonia Audio Detection Models')
    parser.add_argument('--fast', action='store_true', 
                        help='Fast mode: use only a sample of data for quicker training')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of samples to use in fast mode (default: 2000)')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Number of parallel workers (default: {max(1, cpu_count() - 1)} = CPU count - 1)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable feature caching (slower but uses less disk space)')
    
    args = parser.parse_args()
    
    default_workers = max(1, cpu_count() - 1)
    
    print("\n" + "=" * 60)
    print("USAGE OPTIONS:")
    print("=" * 60)
    print("  Normal mode (all data):  python train_pneumonia_audio_models.py")
    print("  Fast mode (2000 samples): python train_pneumonia_audio_models.py --fast")
    print("  Custom samples:          python train_pneumonia_audio_models.py --fast --samples 1000")
    print(f"  More workers:            python train_pneumonia_audio_models.py --workers {cpu_count()}")
    print("  No caching:              python train_pneumonia_audio_models.py --no-cache")
    print("=" * 60)
    
    train_audio_models(
        fast_mode=args.fast,
        sample_size=args.samples,
        workers=args.workers,
        no_cache=args.no_cache
    )

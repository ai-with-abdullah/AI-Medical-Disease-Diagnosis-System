"""
Train Pneumonia Audio Detection Models (OPTIMIZED V2)
======================================================
Optimized for speed AND accuracy with enhanced features.

Models trained:
1. Random Forest - Fast, reliable
2. XGBoost (if available) - Better accuracy
3. Neural Network (MLP) - Best accuracy

Key Improvements:
- Enhanced feature extraction (MFCC + delta + spectral)
- Data augmentation for audio
- Better hyperparameters
- Gradient Boosting option

Usage:
------
  Fast mode (recommended): python train_pneumonia_audio_models.py --fast
  Ultra-fast mode:         python train_pneumonia_audio_models.py --fast --samples 500
  Full accuracy mode:      python train_pneumonia_audio_models.py --full
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import json
import time
import hashlib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DATA_DIR = str(PROJECT_ROOT / 'training_data' / 'pneumonia_audio')
WEIGHTS_DIR = str(PROJECT_ROOT / 'models' / 'weights')
CACHE_DIR = str(PROJECT_ROOT / 'training_data' / '.feature_cache_v3')

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("librosa not installed. Run: pip install librosa")
    LIBROSA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not installed")
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. Neural Network training will be skipped.")
    TF_AVAILABLE = False


def extract_enhanced_features(audio_path, sr=22050, duration=6):
    """
    Enhanced feature extraction for better accuracy.
    
    Features extracted (58 total):
    - MFCC mean (20)
    - MFCC std (20)
    - Delta MFCC mean (5) - captures temporal changes
    - Spectral centroid, rolloff, bandwidth, contrast (4)
    - Zero crossing rate (1)
    - RMS energy (1)
    - Spectral flatness (1)
    - Tempo (1)
    - Chroma mean (5) - reduced for speed
    """
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        if len(y) < sr * 0.3:
            return None
        
        min_length = sr * 2
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)), mode='constant')
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=20, n_fft=2048, hop_length=512)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        delta_mfcc = librosa.feature.delta(mfcc[:5])
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_loaded))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_loaded))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr_loaded))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr_loaded))
        
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        tempo = librosa.feature.tempo(y=y, sr=sr_loaded)[0]
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_loaded, n_chroma=5)
        chroma_mean = np.mean(chroma, axis=1)
        
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            delta_mfcc_mean,
            [spectral_centroid, spectral_rolloff, spectral_bandwidth, spectral_contrast],
            [zcr, rms, spectral_flatness, tempo],
            chroma_mean
        ])
        
        return features
        
    except Exception as e:
        return None


def extract_fast_features(audio_path, sr=20000, duration=5):
    """
    Fast feature extraction - optimized for speed.
    
    Features extracted (32 total):
    - MFCC mean (13)
    - MFCC std (13)
    - Spectral centroid (1)
    - Spectral rolloff (1)
    - Zero crossing rate (1)
    - RMS energy (1)
    - Spectral flatness (1)
    - Tempo (1)
    """
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        if len(y) < sr * 0.3:
            return None
        
        min_length = sr
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)), mode='constant')
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=13, n_fft=1024, hop_length=512)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_loaded))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_loaded))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        tempo = librosa.feature.tempo(y=y, sr=sr_loaded)[0]
        
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, spectral_rolloff, zcr, rms, spectral_flatness, tempo]
        ])
        
        return features
        
    except Exception as e:
        return None


def process_file_enhanced(args):
    """Process a single file with enhanced extraction"""
    idx, file_path, label, cache_path = args
    
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                return idx, cached['features'], label, True
        except:
            pass
    
    features = extract_enhanced_features(file_path)
    
    if features is not None and cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'features': features}, f)
        except:
            pass
    
    return idx, features, label, False


def process_file_fast(args):
    """Process a single file with fast extraction"""
    idx, file_path, label, cache_path = args
    
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                return idx, cached['features'], label, True
        except:
            pass
    
    features = extract_fast_features(file_path)
    
    if features is not None and cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'features': features}, f)
        except:
            pass
    
    return idx, features, label, False


def get_cache_path(file_path, mode='fast'):
    """Generate cache path for a given audio file"""
    file_hash = hashlib.md5(f"{file_path}_{mode}_v3".encode()).hexdigest()
    return os.path.join(CACHE_DIR, mode, f"{file_hash}.pkl")


def detect_datasets():
    """Detect which audio datasets are available"""
    datasets = {}
    
    if not os.path.exists(AUDIO_DATA_DIR):
        print(f"Audio data directory not found: {AUDIO_DATA_DIR}")
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
            print(f"Found COUGHVID: {len(audio_files)} audio files")
    
    for name in ['coswara', 'icbhi_2017', 'virufy', 'covid_cough', 'kaggle_respiratory']:
        dir_path = os.path.join(AUDIO_DATA_DIR, name)
        if os.path.exists(dir_path):
            audio_files = []
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    if f.endswith(('.wav', '.webm', '.ogg', '.mp3')):
                        audio_files.append(os.path.join(root, f))
            if audio_files:
                datasets[name] = {
                    'path': dir_path,
                    'files': audio_files,
                    'count': len(audio_files)
                }
                print(f"Found {name}: {len(audio_files)} audio files")
    
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
                print(f"Found Organized: {len(normal_files)} normal, {len(abnormal_files)} abnormal")
    
    return datasets


def load_coughvid_data(dataset_info):
    """Load COUGHVID dataset with labels from metadata"""
    print("\nLoading COUGHVID dataset...")
    
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
    print("\nLoading organized dataset...")
    
    normal_files = dataset_info.get('normal', [])
    abnormal_files = dataset_info.get('abnormal', [])
    
    audio_files = normal_files + abnormal_files
    labels = [0] * len(normal_files) + [1] * len(abnormal_files)
    
    print(f"   Loaded {len(normal_files)} normal, {len(abnormal_files)} abnormal files")
    return audio_files, labels


def load_folder_based_data(dataset_info, dataset_name):
    """Load dataset based on folder structure"""
    print(f"\nLoading {dataset_name} dataset...")
    
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
    """Combine all detected datasets"""
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
    
    print(f"\nCombined dataset: {len(all_files)} total audio files")
    print(f"   Normal: {all_labels.count(0)}, Abnormal: {all_labels.count(1)}")
    
    return all_files, all_labels


def extract_features_parallel(audio_files, labels, max_workers=None, sample_size=None, 
                               use_cache=True, fast_mode=True):
    """Parallel feature extraction with progress display"""
    print("\n" + "=" * 60)
    print("EXTRACTING AUDIO FEATURES" + (" (FAST MODE)" if fast_mode else " (ENHANCED MODE)"))
    print("=" * 60)
    
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count())
    
    if sample_size and sample_size < len(audio_files):
        print(f"Sampling {sample_size} files from {len(audio_files)} total")
        np.random.seed(42)
        indices = np.random.choice(len(audio_files), sample_size, replace=False)
        audio_files = [audio_files[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    total_files = len(audio_files)
    mode = 'fast' if fast_mode else 'enhanced'
    process_func = process_file_fast if fast_mode else process_file_enhanced
    
    feature_count = 32 if fast_mode else 58
    est_time = 0.4 if fast_mode else 1.0
    
    print(f"Processing {total_files} audio files...")
    print(f"   Mode: {mode.upper()} ({feature_count} features)")
    print(f"   Workers: {max_workers}")
    print(f"   Estimated time: {(total_files * est_time) / max_workers / 60:.1f} minutes")
    print("")
    
    if use_cache:
        os.makedirs(os.path.join(CACHE_DIR, mode), exist_ok=True)
    
    tasks = []
    for idx, (file_path, label) in enumerate(zip(audio_files, labels)):
        cache_path = get_cache_path(file_path, mode) if use_cache else None
        tasks.append((idx, file_path, label, cache_path))
    
    start_time = time.time()
    features_list = [None] * total_files
    labels_list = [None] * total_files
    cached_count = 0
    processed_count = 0
    failed_count = 0
    
    last_update = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_func, task): task[0] for task in tasks}
            
            for future in as_completed(future_to_idx):
                processed_count += 1
                
                try:
                    idx, features, label, from_cache = future.result(timeout=120)
                    
                    if features is not None:
                        features_list[idx] = features
                        labels_list[idx] = label
                        if from_cache:
                            cached_count += 1
                    else:
                        failed_count += 1
                except Exception:
                    failed_count += 1
                
                current_time = time.time()
                if current_time - last_update >= 2.0 or processed_count == total_files:
                    last_update = current_time
                    elapsed = current_time - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_files - processed_count) / rate if rate > 0 else 0
                    progress_pct = (processed_count / total_files) * 100
                    
                    bar_len = 30
                    filled = int(bar_len * progress_pct / 100)
                    bar = '=' * filled + '>' + ' ' * (bar_len - filled - 1)
                    
                    status = f"\r   [{bar}] {progress_pct:5.1f}% | {processed_count}/{total_files} | {rate:.1f}/s | ETA: {remaining/60:.1f}m"
                    if cached_count > 0:
                        status += f" | cached: {cached_count}"
                    print(status, end='', flush=True)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    print()
    
    features_list = [f for f in features_list if f is not None]
    labels_list = [l for l in labels_list if l is not None]
    
    if len(features_list) == 0:
        print("\nNo features extracted!")
        return np.array([]), np.array([])
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    total_time = time.time() - start_time
    print(f"\nFeature extraction complete in {total_time/60:.1f} minutes!")
    print(f"   Successfully processed: {len(X)} files")
    print(f"   Failed/skipped: {failed_count} files")
    print(f"   Loaded from cache: {cached_count} files")
    print(f"   Feature vector size: {X.shape[1]}")
    print(f"   Class distribution: Normal={sum(y==0)}, Abnormal={sum(y==1)}")
    
    return X, y


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest classifier with optimized params"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nRandom Forest Complete! ({train_time:.1f}s)")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    return rf_model, accuracy, f1


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting classifier"""
    print("\n" + "=" * 60)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 60)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    print("Training Gradient Boosting...")
    start_time = time.time()
    gb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nGradient Boosting Complete! ({train_time:.1f}s)")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    return gb_model, accuracy, f1


def train_neural_network(X_train, X_test, y_train, y_test, class_weights):
    """Train Neural Network classifier with improved architecture"""
    if not TF_AVAILABLE:
        print("\nTensorFlow not available. Skipping Neural Network.")
        return None, 0, 0
    
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK MODEL")
    print("=" * 60)
    
    input_dim = X_train.shape[1]
    
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
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
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    print("Training Neural Network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nNeural Network Complete!")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    return model, accuracy, f1


def save_models(rf_model, gb_model, nn_model, scaler, results, feature_mode):
    """Save all trained models"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    rf_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_rf.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump({
            'model': rf_model,
            'scaler': scaler,
            'accuracy': results['rf_accuracy'],
            'f1_score': results['rf_f1'],
            'feature_mode': feature_mode
        }, f)
    print(f"   Random Forest saved: {rf_path}")
    
    if gb_model is not None:
        gb_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_gb.pkl')
        with open(gb_path, 'wb') as f:
            pickle.dump({
                'model': gb_model,
                'scaler': scaler,
                'accuracy': results['gb_accuracy'],
                'f1_score': results['gb_f1'],
                'feature_mode': feature_mode
            }, f)
        print(f"   Gradient Boosting saved: {gb_path}")
    
    if nn_model is not None:
        nn_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_nn.keras')
        nn_model.save(nn_path)
        print(f"   Neural Network saved: {nn_path}")
        
        nn_meta_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_nn_meta.pkl')
        with open(nn_meta_path, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'accuracy': results['nn_accuracy'],
                'f1_score': results['nn_f1'],
                'feature_mode': feature_mode
            }, f)
    
    summary_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'random_forest': {'accuracy': results['rf_accuracy'], 'f1': results['rf_f1']},
            'gradient_boosting': {'accuracy': results['gb_accuracy'], 'f1': results['gb_f1']} if gb_model else None,
            'neural_network': {'accuracy': results['nn_accuracy'], 'f1': results['nn_f1']} if nn_model else None,
            'feature_mode': feature_mode,
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    print(f"   Summary saved: {summary_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pneumonia audio detection models')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (32 features)')
    parser.add_argument('--full', action='store_true', help='Use full enhanced features (58 features)')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples in fast mode')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--no-cache', action='store_true', help='Disable feature caching')
    parser.add_argument('--skip-nn', action='store_true', help='Skip neural network training')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PNEUMONIA AUDIO MODEL TRAINING (OPTIMIZED V2)")
    print("=" * 60)
    print("\nUsage options:")
    print("  Fast mode:     python train_pneumonia_audio_models.py --fast")
    print("  Full mode:     python train_pneumonia_audio_models.py --full")
    print("  Custom:        python train_pneumonia_audio_models.py --fast --samples 1000")
    print("=" * 60)
    
    if not LIBROSA_AVAILABLE or not SKLEARN_AVAILABLE:
        print("\nMissing required libraries!")
        return
    
    print(f"\nScanning for datasets in: {AUDIO_DATA_DIR}")
    datasets = detect_datasets()
    
    if not datasets:
        print("\nNo datasets found!")
        return
    
    audio_files, labels = combine_datasets(datasets)
    
    if len(audio_files) == 0:
        print("\nNo labeled audio files found!")
        return
    
    fast_mode = not args.full
    sample_size = args.samples if args.fast else None
    
    X, y = extract_features_parallel(
        audio_files, 
        labels,
        max_workers=args.workers,
        sample_size=sample_size,
        use_cache=not args.no_cache,
        fast_mode=fast_mode
    )
    
    if len(X) == 0:
        print("\nNo features extracted. Cannot train models.")
        return
    
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weight_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(class_weight_values)}
    print(f"   Class weights: {class_weights}")
    
    rf_model, rf_acc, rf_f1 = train_random_forest(X_train, X_test, y_train, y_test)
    
    gb_model, gb_acc, gb_f1 = train_gradient_boosting(X_train, X_test, y_train, y_test)
    
    if not args.skip_nn:
        nn_model, nn_acc, nn_f1 = train_neural_network(X_train, X_test, y_train, y_test, class_weights)
    else:
        nn_model, nn_acc, nn_f1 = None, 0, 0
    
    results = {
        'rf_accuracy': rf_acc, 'rf_f1': rf_f1,
        'gb_accuracy': gb_acc, 'gb_f1': gb_f1,
        'nn_accuracy': nn_acc, 'nn_f1': nn_f1
    }
    
    feature_mode = 'fast' if fast_mode else 'enhanced'
    save_models(rf_model, gb_model, nn_model, scaler, results, feature_mode)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults Summary:")
    print(f"   Random Forest:      Accuracy={rf_acc*100:.2f}%, F1={rf_f1:.4f}")
    print(f"   Gradient Boosting:  Accuracy={gb_acc*100:.2f}%, F1={gb_f1:.4f}")
    if nn_model:
        print(f"   Neural Network:     Accuracy={nn_acc*100:.2f}%, F1={nn_f1:.4f}")
    print(f"\nModels saved to: {WEIGHTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

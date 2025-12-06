"""
Train Pneumonia Audio Detection Models (OPTIMIZED VERSION)
===========================================================
Highly optimized audio classification for pneumonia detection.
Uses simplified feature extraction for 10-20x faster training.

Models trained:
1. Random Forest - Fast, reliable baseline
2. Neural Network (MLP) - Higher accuracy

Usage:
------
  Fast mode (recommended): python train_pneumonia_audio_models.py --fast
  Ultra-fast mode:         python train_pneumonia_audio_models.py --fast --samples 500
  Normal mode:             python train_pneumonia_audio_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pickle
import json
import time
import hashlib
from functools import partial

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_DATA_DIR = str(PROJECT_ROOT / 'training_data' / 'pneumonia_audio')
WEIGHTS_DIR = str(PROJECT_ROOT / 'models' / 'weights')
CACHE_DIR = str(PROJECT_ROOT / 'training_data' / '.feature_cache_v2')

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("librosa not installed. Run: pip install librosa")
    LIBROSA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not installed. Run: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. Neural Network training will be skipped.")
    TF_AVAILABLE = False


def extract_fast_features(audio_path, sr=16000, duration=5):
    """
    FAST feature extraction - simplified for speed.
    
    Features extracted (26 total):
    - MFCC mean (13) - reduced from 40
    - MFCC std (13)
    
    Speed: ~0.3-0.5 seconds per file vs 3-5 seconds
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
        
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        return features
        
    except Exception as e:
        return None


def extract_standard_features(audio_path, sr=22050, duration=6):
    """
    Standard feature extraction - balanced speed and accuracy.
    
    Features extracted (54 total):
    - MFCC mean (20)
    - MFCC std (20)
    - Spectral centroid (1)
    - Spectral rolloff (1)
    - Spectral bandwidth (1)
    - Zero crossing rate (1)
    - RMS energy (1)
    - Spectral contrast mean (7) - reduced
    - Tempo (1)
    - Spectral flatness (1)
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
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_loaded))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_loaded))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr_loaded))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr_loaded), axis=1)
        
        tempo = librosa.feature.tempo(y=y, sr=sr_loaded)[0]
        
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr, rms],
            spectral_contrast,
            [tempo, spectral_flatness]
        ])
        
        return features
        
    except Exception as e:
        return None


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


def process_file_standard(args):
    """Process a single file with standard extraction"""
    idx, file_path, label, cache_path = args
    
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                return idx, cached['features'], label, True
        except:
            pass
    
    features = extract_standard_features(file_path)
    
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
    file_hash = hashlib.md5(f"{file_path}_{mode}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, mode, f"{file_hash}.pkl")


def detect_datasets():
    """Detect which audio datasets are available"""
    datasets = {}
    
    if not os.path.exists(AUDIO_DATA_DIR):
        print(f"Audio data directory not found: {AUDIO_DATA_DIR}")
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


def extract_features_optimized(audio_files, labels, max_workers=None, sample_size=None, 
                                use_cache=True, fast_mode=True):
    """
    OPTIMIZED feature extraction using ProcessPoolExecutor.
    Much faster than the original multiprocessing.Pool approach.
    """
    print("\n" + "=" * 60)
    print("EXTRACTING AUDIO FEATURES" + (" (FAST MODE)" if fast_mode else ""))
    print("=" * 60)
    
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count())
    
    if sample_size and sample_size < len(audio_files):
        print(f"FAST MODE: Sampling {sample_size} files from {len(audio_files)} total")
        np.random.seed(42)
        indices = np.random.choice(len(audio_files), sample_size, replace=False)
        audio_files = [audio_files[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    total_files = len(audio_files)
    mode = 'fast' if fast_mode else 'standard'
    process_func = process_file_fast if fast_mode else process_file_standard
    
    est_time_per_file = 0.5 if fast_mode else 1.5
    est_total_time = (total_files * est_time_per_file) / max_workers
    
    print(f"Processing {total_files} audio files...")
    print(f"   Mode: {'FAST (26 features)' if fast_mode else 'STANDARD (54 features)'}")
    print(f"   Workers: {max_workers}")
    print(f"   Estimated time: {est_total_time/60:.1f} minutes")
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
    valid_count = 0
    
    last_update = time.time()
    update_interval = 2.0
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            chunk_size = max(1, total_files // (max_workers * 4))
            future_to_idx = {executor.submit(process_func, task): task[0] for task in tasks}
            
            for future in as_completed(future_to_idx):
                processed_count += 1
                
                try:
                    idx, features, label, from_cache = future.result(timeout=60)
                    
                    if features is not None:
                        features_list[idx] = features
                        labels_list[idx] = label
                        valid_count += 1
                        if from_cache:
                            cached_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                
                current_time = time.time()
                if current_time - last_update >= update_interval or processed_count == total_files:
                    last_update = current_time
                    elapsed = current_time - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_files - processed_count) / rate if rate > 0 else 0
                    progress_pct = (processed_count / total_files) * 100
                    
                    bar_length = 30
                    filled = int(bar_length * progress_pct / 100)
                    bar = '=' * filled + '>' + ' ' * (bar_length - filled - 1)
                    
                    status = f"\r   [{bar}] {progress_pct:5.1f}% | {processed_count}/{total_files} | {rate:.1f}/s | ETA: {remaining/60:.1f}m"
                    if cached_count > 0:
                        status += f" | cached: {cached_count}"
                    print(status, end='', flush=True)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"   Processed {processed_count} files before interrupt")
    
    print()
    
    features_list = [f for f in features_list if f is not None]
    labels_list = [l for l in labels_list if l is not None]
    
    if len(features_list) == 0:
        print("\nNo features extracted! Check your audio files.")
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


def train_random_forest(X_train, X_test, y_train, y_test, class_weights):
    """Train Random Forest classifier"""
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("Training Random Forest...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRandom Forest Training Complete! ({train_time:.1f}s)")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    return rf_model, accuracy


def train_neural_network(X_train, X_test, y_train, y_test, class_weights):
    """Train Neural Network (MLP) classifier"""
    if not TF_AVAILABLE:
        print("\nTensorFlow not available. Skipping Neural Network training.")
        return None, 0
    
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK MODEL")
    print("=" * 60)
    
    input_dim = X_train.shape[1]
    
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
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
            patience=8,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    print("Training Neural Network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nNeural Network Training Complete!")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    
    return model, accuracy


def save_models(rf_model, nn_model, scaler, accuracy_rf, accuracy_nn, feature_mode):
    """Save trained models to weights directory"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    rf_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_rf.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump({
            'model': rf_model,
            'scaler': scaler,
            'accuracy': accuracy_rf,
            'feature_mode': feature_mode,
            'feature_count': 26 if feature_mode == 'fast' else 54
        }, f)
    print(f"   Random Forest saved: {rf_path}")
    
    if nn_model is not None:
        nn_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_nn.keras')
        nn_model.save(nn_path)
        print(f"   Neural Network saved: {nn_path}")
        
        nn_meta_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_nn_meta.pkl')
        with open(nn_meta_path, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'accuracy': accuracy_nn,
                'feature_mode': feature_mode,
                'feature_count': 26 if feature_mode == 'fast' else 54
            }, f)
        print(f"   Neural Network metadata saved: {nn_meta_path}")
    
    summary = {
        'random_forest': {
            'path': 'pneumonia_audio_rf.pkl',
            'accuracy': float(accuracy_rf),
            'feature_mode': feature_mode
        },
        'neural_network': {
            'path': 'pneumonia_audio_nn.keras' if nn_model else None,
            'accuracy': float(accuracy_nn) if nn_model else None,
            'feature_mode': feature_mode
        },
        'feature_mode': feature_mode,
        'feature_count': 26 if feature_mode == 'fast' else 54,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_path = os.path.join(WEIGHTS_DIR, 'pneumonia_audio_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Training summary saved: {summary_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pneumonia audio detection models')
    parser.add_argument('--fast', action='store_true', help='Use fast mode with fewer samples')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples in fast mode')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--no-cache', action='store_true', help='Disable feature caching')
    parser.add_argument('--standard-features', action='store_true', help='Use standard features (slower but more accurate)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("USAGE OPTIONS:")
    print("=" * 60)
    print("  Fast mode (recommended):  python train_pneumonia_audio_models.py --fast")
    print("  Ultra-fast mode:          python train_pneumonia_audio_models.py --fast --samples 500")
    print("  More workers:             python train_pneumonia_audio_models.py --workers 8")
    print("  Standard features:        python train_pneumonia_audio_models.py --standard-features")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("PNEUMONIA AUDIO MODEL TRAINING (OPTIMIZED)")
    print("=" * 60)
    if args.fast:
        print(f"FAST MODE ENABLED - Using {args.samples} samples")
    if args.standard_features:
        print("STANDARD FEATURES - More features, slower processing")
    else:
        print("FAST FEATURES - 26 features, faster processing")
    print("=" * 60)
    
    if not LIBROSA_AVAILABLE or not SKLEARN_AVAILABLE:
        print("\nMissing required libraries!")
        return
    
    print(f"\nScanning for datasets in: {AUDIO_DATA_DIR}")
    datasets = detect_datasets()
    
    if not datasets:
        print("\nNo datasets found!")
        print("Please download audio datasets and place them in:")
        print(f"   {AUDIO_DATA_DIR}/")
        print("\nSupported datasets:")
        print("   - coughvid/ (COUGHVID dataset)")
        print("   - organized/ with normal/ and abnormal/ subfolders")
        return
    
    audio_files, labels = combine_datasets(datasets)
    
    if len(audio_files) == 0:
        print("\nNo labeled audio files found!")
        return
    
    sample_size = args.samples if args.fast else None
    fast_mode = not args.standard_features
    
    X, y = extract_features_optimized(
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
    
    rf_model, accuracy_rf = train_random_forest(X_train, X_test, y_train, y_test, class_weights)
    
    nn_model, accuracy_nn = train_neural_network(X_train, X_test, y_train, y_test, class_weights)
    
    feature_mode = 'fast' if fast_mode else 'standard'
    save_models(rf_model, nn_model, scaler, accuracy_rf, accuracy_nn, feature_mode)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Random Forest Accuracy: {accuracy_rf*100:.2f}%")
    if nn_model:
        print(f"   Neural Network Accuracy: {accuracy_nn*100:.2f}%")
    print(f"\nModels saved to: {WEIGHTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

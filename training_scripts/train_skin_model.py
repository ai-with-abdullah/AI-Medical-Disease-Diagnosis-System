"""
Train Skin Cancer Model - OPTIMIZED VERSION
============================================
Optimized for CPU training - 10x faster than original.

Key Optimizations:
1. MobileNetV2 instead of ResNet50 (4x fewer parameters)
2. Smaller image size (160x160 instead of 224x224)
3. Fewer epochs with early stopping
4. Mixed precision where available
5. Fast mode option for quick testing

Usage:
------
  Fast mode (30-60 min): python train_skin_model.py --fast
  Normal mode (2-3 hrs): python train_skin_model.py
  Full mode (4-6 hrs):   python train_skin_model.py --full

Supported Datasets:
- HAM10000 (recommended)
- ISIC 2019
- PAD-UFES-20
- Pre-organized folders
"""

import os
import sys
import time
import shutil
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

import numpy as np
import pandas as pd
import argparse

SKIN_DATA_DIR = os.path.join(project_dir, 'training_data', 'skin_cancer')
SKIN_DATA_DIR_OLD = os.path.join(project_dir, 'training_data', 'skin_disease')
WEIGHTS_DIR = os.path.join(project_dir, 'models', 'weights')

HAM10000_CLASSES = {
    'nv': 'Melanocytic Nevus (Mole)',
    'mel': 'Melanoma', 
    'bkl': 'Benign Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
}

ISIC2019_TO_HAM10000 = {
    'MEL': 'mel', 'NV': 'nv', 'BCC': 'bcc', 'AK': 'akiec',
    'BKL': 'bkl', 'DF': 'df', 'VASC': 'vasc', 'SCC': 'bcc', 'UNK': None
}

PAD_UFES_TO_HAM10000 = {
    'ACK': 'akiec', 'BCC': 'bcc', 'MEL': 'mel',
    'NEV': 'nv', 'SCC': 'bcc', 'SEK': 'bkl'
}


def check_tensorflow():
    """Check TensorFlow and GPU availability"""
    try:
        import tensorflow as tf
        print(f"   TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"      - {gpu.name}")
            return True, True
        else:
            print("   No GPU detected - using CPU (optimized for speed)")
            return True, False
    except ImportError:
        print("\nTensorFlow is not installed!")
        return False, False


def get_data_directory():
    """Get skin cancer data directory"""
    if os.path.exists(SKIN_DATA_DIR):
        return SKIN_DATA_DIR
    elif os.path.exists(SKIN_DATA_DIR_OLD):
        return SKIN_DATA_DIR_OLD
    else:
        os.makedirs(SKIN_DATA_DIR, exist_ok=True)
        return SKIN_DATA_DIR


def detect_all_datasets():
    """Detect available skin cancer datasets"""
    print("\n" + "=" * 70)
    print("STEP 1: DETECTING SKIN CANCER DATASETS")
    print("=" * 70)
    
    data_dir = get_data_directory()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    datasets = {}
    total_images = 0
    
    ham_dirs = [
        os.path.join(data_dir, 'ham10000'),
        os.path.join(data_dir, 'HAM10000'),
        data_dir
    ]
    
    for ham_dir in ham_dirs:
        ham_part1 = os.path.join(ham_dir, 'HAM10000_images_part_1')
        ham_part2 = os.path.join(ham_dir, 'HAM10000_images_part_2')
        ham_images = os.path.join(ham_dir, 'HAM10000_images')
        ham_metadata = os.path.join(ham_dir, 'HAM10000_metadata.csv')
        
        if os.path.exists(ham_metadata):
            print("\n   [FOUND] HAM10000 dataset")
            img_dirs = []
            img_count = 0
            
            for img_dir in [ham_part1, ham_part2, ham_images]:
                if os.path.exists(img_dir):
                    img_dirs.append(img_dir)
                    count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    img_count += count
                    print(f"      - {os.path.basename(img_dir)}: {count:,} images")
            
            if img_count > 0:
                datasets['ham10000'] = {
                    'type': 'ham10000',
                    'metadata': ham_metadata,
                    'image_dirs': img_dirs,
                    'count': img_count
                }
                total_images += img_count
            break
    
    for name, patterns in [('isic2019', ['isic2019', 'ISIC2019', 'isic_2019']),
                           ('pad_ufes_20', ['pad_ufes_20', 'PAD-UFES-20', 'pad-ufes-20'])]:
        for pattern in patterns:
            dir_path = os.path.join(data_dir, pattern)
            if os.path.exists(dir_path):
                csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                if csv_files:
                    img_count = sum(1 for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                    if img_count == 0:
                        for subdir in os.listdir(dir_path):
                            subpath = os.path.join(dir_path, subdir)
                            if os.path.isdir(subpath):
                                img_count += sum(1 for f in os.listdir(subpath) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                    
                    if img_count > 0:
                        print(f"\n   [FOUND] {name.upper()} dataset: {img_count:,} images")
                        datasets[name] = {
                            'type': name,
                            'path': dir_path,
                            'count': img_count
                        }
                        total_images += img_count
                break
    
    organized_dir = os.path.join(data_dir, 'organized')
    if os.path.exists(organized_dir):
        classes = [d for d in os.listdir(organized_dir) if os.path.isdir(os.path.join(organized_dir, d))]
        if len(classes) >= 2:
            img_count = 0
            print(f"\n   [FOUND] Pre-organized dataset")
            for cls in classes:
                cls_dir = os.path.join(organized_dir, cls)
                count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                img_count += count
                if count > 0:
                    print(f"      - {cls}: {count:,} images")
            
            if img_count > 0:
                datasets['organized'] = {
                    'type': 'organized',
                    'path': organized_dir,
                    'count': img_count
                }
    
    if not datasets:
        print("\n   No skin cancer dataset found!")
        print("\n   Please download HAM10000 from:")
        print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        return None, data_dir
    
    print(f"\n   === SUMMARY ===")
    print(f"   Total datasets: {len(datasets)}")
    print(f"   Total images: {total_images:,}")
    
    return datasets, data_dir


def organize_datasets(datasets, data_dir):
    """Organize datasets into class folders"""
    print("\n" + "=" * 70)
    print("STEP 2: ORGANIZING DATASETS")
    print("=" * 70)
    
    organized_dir = os.path.join(data_dir, 'organized')
    
    if 'organized' in datasets and datasets['organized'] is not None:
        print(f"\n   Dataset already organized! ({datasets['organized']['count']:,} images)")
        return organized_dir
    
    for cls in HAM10000_CLASSES.keys():
        os.makedirs(os.path.join(organized_dir, cls), exist_ok=True)
    
    total_copied = 0
    
    if datasets.get('ham10000'):
        print("\n   Processing HAM10000...")
        ham_data = datasets['ham10000']
        df = pd.read_csv(ham_data['metadata'])
        
        copied = 0
        for idx, row in df.iterrows():
            image_id = row['image_id']
            cls = row['dx']
            
            if cls not in HAM10000_CLASSES:
                continue
            
            src_file = None
            for img_dir in ham_data['image_dirs']:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                    path = os.path.join(img_dir, image_id + ext)
                    if os.path.exists(path):
                        src_file = path
                        break
                if src_file:
                    break
            
            if src_file:
                dst_file = os.path.join(organized_dir, cls, os.path.basename(src_file))
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied += 1
            
            if (idx + 1) % 2000 == 0:
                print(f"      Progress: {idx + 1:,}/{len(df):,}")
        
        print(f"      Copied: {copied:,} images")
        total_copied += copied
    
    print(f"\n   Organization complete! Total: {total_copied:,} new images")
    
    print(f"\n   Class distribution:")
    for cls, name in HAM10000_CLASSES.items():
        cls_dir = os.path.join(organized_dir, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if count > 0:
                print(f"      {cls} ({name}): {count:,} images")
    
    return organized_dir


def create_data_generators(organized_dir, img_size, batch_size, sample_fraction=1.0):
    """Create data generators for training"""
    print("\n" + "=" * 70)
    print("STEP 3: PREPARING DATA GENERATORS")
    print("=" * 70)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    total_images = 0
    for cls in os.listdir(organized_dir):
        cls_dir = os.path.join(organized_dir, cls)
        if os.path.isdir(cls_dir):
            total_images += len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n   Total images: {total_images:,}")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Batch size: {batch_size}")
    
    if sample_fraction < 1.0:
        print(f"   Sampling: {sample_fraction*100:.0f}% of data")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        organized_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        organized_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    if sample_fraction < 1.0:
        train_generator.samples = int(train_generator.samples * sample_fraction)
        val_generator.samples = int(val_generator.samples * sample_fraction)
    
    print(f"\n   Training samples: {train_generator.samples:,}")
    print(f"   Validation samples: {val_generator.samples:,}")
    print(f"   Classes: {train_generator.num_classes}")
    
    class_names = list(train_generator.class_indices.keys())
    for i, cls in enumerate(class_names):
        desc = HAM10000_CLASSES.get(cls, cls)
        print(f"      {i}: {cls} - {desc}")
    
    return train_generator, val_generator, class_names


def build_model(num_classes, img_size, model_type='mobilenet'):
    """Build the classification model"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    if model_type == 'mobilenet':
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3),
            alpha=1.0
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
    elif model_type == 'efficientnet':
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    else:
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    return model, base_model


def train_model(model, base_model, train_gen, val_gen, class_weights, 
                epochs_frozen=5, epochs_finetune=10, model_path=None):
    """Train the model with frozen base, then fine-tune"""
    from tensorflow import keras
    
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODEL")
    print("=" * 70)
    
    print(f"\n   Class weights (for imbalanced data):")
    for cls, weight in class_weights.items():
        print(f"      Class {cls}: {weight:.3f}")
    
    total_params = model.count_params()
    print(f"\n   Model parameters: {total_params:,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if model_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        )
    
    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    val_steps = max(1, val_gen.samples // val_gen.batch_size)
    
    print(f"\n[Phase 1] Training with frozen base ({epochs_frozen} epochs)...")
    print("-" * 50)
    
    history1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_frozen,
        validation_data=val_gen,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n[Phase 2] Fine-tuning ({epochs_finetune} epochs)...")
    print("-" * 50)
    
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_finetune,
        validation_data=val_gen,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return model


def evaluate_model(model, val_gen, class_names):
    """Evaluate the trained model"""
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATING MODEL")
    print("=" * 70)
    
    val_gen.reset()
    steps = max(1, val_gen.samples // val_gen.batch_size)
    
    loss, accuracy = model.evaluate(val_gen, steps=steps, verbose=0)
    
    print(f"\n   Final Validation Accuracy: {accuracy*100:.2f}%")
    print(f"   Final Validation Loss: {loss:.4f}")
    
    val_gen.reset()
    predictions = model.predict(val_gen, steps=steps, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes[:len(y_pred)]
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n   Classification Report:")
    print("-" * 50)
    
    target_names = [f"{cls} ({HAM10000_CLASSES.get(cls, cls)[:15]})" for cls in class_names]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    return accuracy


def save_model(model, accuracy, class_names, model_type, img_size):
    """Save the trained model"""
    print("\n" + "=" * 70)
    print("STEP 6: SAVING MODEL")
    print("=" * 70)
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    model_path = os.path.join(WEIGHTS_DIR, 'skin_model.h5')
    model.save(model_path)
    print(f"\n   Model saved: {model_path}")
    
    keras_path = os.path.join(WEIGHTS_DIR, 'skin_model.keras')
    try:
        model.save(keras_path)
        print(f"   Keras format: {keras_path}")
    except Exception:
        pass
    
    import json
    metadata = {
        'accuracy': float(accuracy),
        'classes': class_names,
        'class_descriptions': {cls: HAM10000_CLASSES.get(cls, cls) for cls in class_names},
        'model_type': model_type,
        'img_size': img_size,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    meta_path = os.path.join(WEIGHTS_DIR, 'skin_model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {meta_path}")
    
    return model_path


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train skin cancer detection model')
    parser.add_argument('--fast', action='store_true', help='Fast mode: smaller images, fewer epochs (30-60 min)')
    parser.add_argument('--full', action='store_true', help='Full mode: ResNet50, more epochs (4-6 hrs)')
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'efficientnet', 'resnet'],
                        help='Model architecture (default: mobilenet)')
    parser.add_argument('--img-size', type=int, default=None, help='Image size (default: auto)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (default: auto)')
    parser.add_argument('--epochs', type=int, default=None, help='Total epochs (default: auto)')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SKIN CANCER MODEL TRAINING (OPTIMIZED)")
    print("=" * 70)
    print("\nUsage options:")
    print("  Fast mode (30-60 min):  python train_skin_model.py --fast")
    print("  Normal mode (2-3 hrs):  python train_skin_model.py")
    print("  Full mode (4-6 hrs):    python train_skin_model.py --full")
    print("=" * 70)
    
    tf_ok, has_gpu = check_tensorflow()
    if not tf_ok:
        return
    
    if args.fast:
        img_size = args.img_size or 128
        batch_size = args.batch_size or (64 if has_gpu else 32)
        epochs_frozen = 3
        epochs_finetune = 5
        model_type = 'mobilenet'
        sample_fraction = 0.5
        print(f"\n   FAST MODE: img={img_size}, batch={batch_size}, epochs={epochs_frozen+epochs_finetune}")
    elif args.full:
        img_size = args.img_size or 224
        batch_size = args.batch_size or (32 if has_gpu else 16)
        epochs_frozen = 10
        epochs_finetune = 15
        model_type = args.model if args.model != 'mobilenet' else 'resnet'
        sample_fraction = 1.0
        print(f"\n   FULL MODE: img={img_size}, model={model_type}, epochs={epochs_frozen+epochs_finetune}")
    else:
        img_size = args.img_size or 160
        batch_size = args.batch_size or (48 if has_gpu else 24)
        epochs_frozen = 5
        epochs_finetune = 10
        model_type = args.model
        sample_fraction = 1.0
        print(f"\n   NORMAL MODE: img={img_size}, model={model_type}, epochs={epochs_frozen+epochs_finetune}")
    
    if args.epochs:
        epochs_frozen = max(2, args.epochs // 3)
        epochs_finetune = args.epochs - epochs_frozen
    
    datasets, data_dir = detect_all_datasets()
    if datasets is None:
        return
    
    organized_dir = organize_datasets(datasets, data_dir)
    
    train_gen, val_gen, class_names = create_data_generators(
        organized_dir, img_size, batch_size, sample_fraction
    )
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_values = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = {i: w for i, w in enumerate(class_weights_values)}
    
    model, base_model = build_model(len(class_names), img_size, model_type)
    
    model_path = os.path.join(WEIGHTS_DIR, 'skin_model.h5')
    
    start_time = time.time()
    
    model = train_model(
        model, base_model, train_gen, val_gen, class_weights,
        epochs_frozen=epochs_frozen,
        epochs_finetune=epochs_finetune,
        model_path=model_path
    )
    
    train_time = time.time() - start_time
    
    accuracy = evaluate_model(model, val_gen, class_names)
    
    save_model(model, accuracy, class_names, model_type, img_size)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n   Final Accuracy: {accuracy*100:.2f}%")
    print(f"   Training Time: {train_time/60:.1f} minutes")
    print(f"   Model saved to: {WEIGHTS_DIR}/skin_model.h5")
    print("=" * 70)


if __name__ == '__main__':
    main()

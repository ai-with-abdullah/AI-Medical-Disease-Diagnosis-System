"""
Train Skin Disease Model - ALL-IN-ONE SCRIPT
=============================================
Just download dataset and run this script - no code changes needed!

This script automatically:
1. Detects skin disease datasets in training_data/skin_disease/
2. Organizes images into class folders if needed
3. Trains ResNet50 model with transfer learning
4. Saves trained model to models/weights/

Supported Datasets (place in training_data/skin_disease/):
----------------------------------------------------------
1. HAM10000 (Recommended)
   - Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Size: 2.7 GB, 10,015 images, 7 classes

2. ISIC Archive (Alternative)
   - Link: https://www.isic-archive.com/
   - Pre-organized in class folders

3. DermNet (Alternative)
   - Link: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
   - Pre-organized in class folders

Usage:
------
1. Download HAM10000 from Kaggle
2. Extract to training_data/skin_disease/
3. Run: python training_scripts/train_skin_model.py

That's it! No code changes required.

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
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

SKIN_DATA_DIR = os.path.join(project_dir, 'training_data', 'skin_disease')
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


def check_tensorflow():
    """Check if TensorFlow is available"""
    try:
        import tensorflow as tf
        print(f"   TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"      - {gpu.name}")
        else:
            print("   No GPU detected - training will use CPU (slower)")
        
        return True
    except ImportError:
        print("\n" + "!" * 70)
        print("ERROR: TensorFlow is not installed!")
        print("!" * 70)
        print("\nInstall TensorFlow with:")
        print("   pip install tensorflow")
        print("\nOr for GPU support:")
        print("   pip install tensorflow[and-cuda]")
        return False


def detect_dataset():
    """Detect which skin disease dataset is available"""
    print("\n" + "=" * 70)
    print("STEP 1: DETECTING SKIN DISEASE DATASET")
    print("=" * 70)
    
    os.makedirs(SKIN_DATA_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    dataset_info = {
        'type': None,
        'images_dirs': [],
        'metadata_file': None,
        'organized_dir': None,
        'total_images': 0
    }
    
    ham_part1 = os.path.join(SKIN_DATA_DIR, 'HAM10000_images_part_1')
    ham_part2 = os.path.join(SKIN_DATA_DIR, 'HAM10000_images_part_2')
    ham_images = os.path.join(SKIN_DATA_DIR, 'HAM10000_images')
    ham_metadata = os.path.join(SKIN_DATA_DIR, 'HAM10000_metadata.csv')
    
    if os.path.exists(ham_metadata):
        print("\n   [FOUND] HAM10000 dataset detected!")
        dataset_info['type'] = 'ham10000'
        dataset_info['metadata_file'] = ham_metadata
        
        for img_dir in [ham_part1, ham_part2, ham_images]:
            if os.path.exists(img_dir):
                dataset_info['images_dirs'].append(img_dir)
                img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   - {os.path.basename(img_dir)}: {img_count:,} images")
                dataset_info['total_images'] += img_count
        
        if dataset_info['total_images'] > 0:
            return dataset_info
    
    organized_dir = os.path.join(SKIN_DATA_DIR, 'organized')
    if os.path.exists(organized_dir):
        classes = [d for d in os.listdir(organized_dir) if os.path.isdir(os.path.join(organized_dir, d))]
        if len(classes) >= 2:
            print("\n   [FOUND] Pre-organized dataset detected!")
            dataset_info['type'] = 'organized'
            dataset_info['organized_dir'] = organized_dir
            
            for cls in classes:
                cls_dir = os.path.join(organized_dir, cls)
                img_count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                dataset_info['total_images'] += img_count
                print(f"   - {cls}: {img_count:,} images")
            
            return dataset_info
    
    for folder in os.listdir(SKIN_DATA_DIR):
        folder_path = os.path.join(SKIN_DATA_DIR, folder)
        if os.path.isdir(folder_path) and folder not in ['organized', '__pycache__']:
            subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if len(subfolders) >= 2:
                has_images = False
                for sf in subfolders[:3]:
                    sf_path = os.path.join(folder_path, sf)
                    images = [f for f in os.listdir(sf_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(images) > 10:
                        has_images = True
                        break
                
                if has_images:
                    print(f"\n   [FOUND] Class-organized dataset in {folder}/")
                    dataset_info['type'] = 'organized'
                    dataset_info['organized_dir'] = folder_path
                    
                    for cls in subfolders:
                        cls_dir = os.path.join(folder_path, cls)
                        if os.path.isdir(cls_dir):
                            img_count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            dataset_info['total_images'] += img_count
                            print(f"   - {cls}: {img_count:,} images")
                    
                    return dataset_info
    
    print("\n" + "!" * 70)
    print("ERROR: No skin disease dataset found!")
    print("!" * 70)
    print("\nPlease download the HAM10000 dataset:")
    print("  1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    print("  2. Download the dataset (2.7 GB)")
    print("  3. Extract to training_data/skin_disease/")
    print("\nExpected structure:")
    print("  training_data/skin_disease/")
    print("  ├── HAM10000_images_part_1/")
    print("  ├── HAM10000_images_part_2/")
    print("  └── HAM10000_metadata.csv")
    
    return None


def organize_ham10000(dataset_info):
    """Organize HAM10000 images into class folders"""
    print("\n" + "=" * 70)
    print("STEP 2: ORGANIZING DATASET BY CLASS")
    print("=" * 70)
    
    organized_dir = os.path.join(SKIN_DATA_DIR, 'organized')
    
    if os.path.exists(organized_dir):
        existing_classes = [d for d in os.listdir(organized_dir) if os.path.isdir(os.path.join(organized_dir, d))]
        total_existing = 0
        for cls in existing_classes:
            cls_dir = os.path.join(organized_dir, cls)
            total_existing += len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if total_existing >= dataset_info['total_images'] * 0.9:
            print(f"\n   Dataset already organized! ({total_existing:,} images)")
            dataset_info['organized_dir'] = organized_dir
            dataset_info['type'] = 'organized'
            return dataset_info
    
    print("\n   Loading metadata...")
    df = pd.read_csv(dataset_info['metadata_file'])
    print(f"   Total records: {len(df):,}")
    
    print("\n   Class distribution:")
    class_counts = df['dx'].value_counts()
    for cls, count in class_counts.items():
        name = HAM10000_CLASSES.get(cls, cls)
        print(f"   - {cls} ({name}): {count:,} images")
    
    for cls in HAM10000_CLASSES.keys():
        cls_dir = os.path.join(organized_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
    
    print("\n   Organizing images by class...")
    copied = 0
    not_found = 0
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        cls = row['dx']
        
        if cls not in HAM10000_CLASSES:
            continue
        
        src_file = None
        for img_dir in dataset_info['images_dirs']:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
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
        else:
            not_found += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"      Progress: {idx + 1:,}/{len(df):,} images")
    
    print(f"\n   Organization complete!")
    print(f"   - Copied: {copied:,} images")
    if not_found > 0:
        print(f"   - Not found: {not_found:,} images")
    
    dataset_info['organized_dir'] = organized_dir
    dataset_info['type'] = 'organized'
    return dataset_info


def create_data_generators(organized_dir, batch_size=32, img_size=(224, 224)):
    """Create training and validation data generators"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        organized_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        organized_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def compute_class_weights(train_generator):
    """Compute class weights for imbalanced dataset"""
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    
    return dict(enumerate(class_weights))


def build_model(num_classes=7):
    """Build ResNet50-based skin disease classification model"""
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def train_skin_model():
    """Main training function"""
    print("\n" + "=" * 70)
    print("SKIN DISEASE MODEL TRAINING - ALL-IN-ONE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Auto-detect skin disease dataset")
    print("  2. Organize images by class (if needed)")
    print("  3. Train ResNet50 model with transfer learning")
    print("  4. Save model to models/weights/skin_resnet50.h5")
    
    if not check_tensorflow():
        return False
    
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    dataset_info = detect_dataset()
    if dataset_info is None:
        return False
    
    if dataset_info['type'] == 'ham10000':
        dataset_info = organize_ham10000(dataset_info)
    
    organized_dir = dataset_info['organized_dir']
    
    print("\n" + "=" * 70)
    print("STEP 3: PREPARING DATA GENERATORS")
    print("=" * 70)
    
    total_images = dataset_info['total_images']
    if total_images > 5000:
        batch_size = 32
    elif total_images > 1000:
        batch_size = 16
    else:
        batch_size = 8
    
    print(f"\n   Total images: {total_images:,}")
    print(f"   Batch size: {batch_size}")
    
    train_gen, val_gen = create_data_generators(organized_dir, batch_size=batch_size)
    
    print(f"\n   Training samples: {train_gen.samples:,}")
    print(f"   Validation samples: {val_gen.samples:,}")
    print(f"   Classes: {len(train_gen.class_indices)}")
    
    for cls, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        name = HAM10000_CLASSES.get(cls, cls)
        print(f"      {idx}: {cls} - {name}")
    
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODEL")
    print("=" * 70)
    
    class_weights = compute_class_weights(train_gen)
    print("\n   Class weights (for imbalanced data):")
    for cls_idx, weight in sorted(class_weights.items()):
        print(f"      Class {cls_idx}: {weight:.3f}")
    
    num_classes = len(train_gen.class_indices)
    model, base_model = build_model(num_classes=num_classes)
    
    print(f"\n   Model architecture:")
    print(f"      Base: ResNet50 (ImageNet weights)")
    print(f"      Output classes: {num_classes}")
    print(f"      Total parameters: {model.count_params():,}")
    
    model_path = os.path.join(WEIGHTS_DIR, 'skin_resnet50.h5')
    
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n[Phase 1] Training with frozen base model...")
    print("-" * 50)
    
    start_time = time.time()
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    phase1_time = time.time() - start_time
    phase1_acc = max(history1.history['val_accuracy'])
    print(f"\n   Phase 1 complete: {phase1_time:.1f}s")
    print(f"   Best validation accuracy: {phase1_acc*100:.2f}%")
    
    print("\n[Phase 2] Fine-tuning with unfrozen layers...")
    print("-" * 50)
    
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_count = sum([layer.trainable for layer in base_model.layers])
    print(f"   Trainable layers: {trainable_count}")
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("STEP 5: RESULTS")
    print("=" * 70)
    
    best_acc = max(max(history1.history['val_accuracy']), max(history2.history['val_accuracy']))
    final_acc = history2.history['val_accuracy'][-1]
    
    print(f"\n   === RESULTS ===")
    print(f"   Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"   Final Validation Accuracy: {final_acc*100:.2f}%")
    print(f"   Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    class_indices = train_gen.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    import json
    class_info_path = os.path.join(WEIGHTS_DIR, 'skin_classes.json')
    with open(class_info_path, 'w') as f:
        json.dump({
            'class_indices': class_indices,
            'class_names': class_names,
            'ham10000_mapping': HAM10000_CLASSES
        }, f, indent=2)
    
    print(f"\n   === SAVED FILES ===")
    print(f"   {model_path}")
    print(f"   {class_info_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n   Model: skin_resnet50.h5")
    print(f"   Accuracy: {best_acc*100:.2f}%")
    print(f"   Classes: {num_classes}")
    
    print(f"\n   === CLASSES TRAINED ===")
    for cls, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        name = HAM10000_CLASSES.get(cls, cls)
        print(f"   {idx+1}. {name}")
    
    print(f"\n   === NEXT STEPS ===")
    print("   1. Restart the Streamlit app")
    print("   2. The app will auto-detect the trained model")
    print("   3. Test with skin lesion images!")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Your skin disease model is ready.")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    train_skin_model()

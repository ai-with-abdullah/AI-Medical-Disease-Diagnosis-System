"""
Train Skin Disease Detection Model
===================================
This script trains a ResNet50 CNN model for skin disease detection.

Model: ResNet50 with transfer learning
Dataset: HAM10000 (7 classes of skin lesions)

Classes:
1. nv - Melanocytic Nevus (Mole)
2. mel - Melanoma
3. bkl - Benign Keratosis
4. bcc - Basal Cell Carcinoma
5. akiec - Actinic Keratosis
6. vasc - Vascular Lesion
7. df - Dermatofibroma

Usage:
------
1. Download HAM10000 dataset from Kaggle
2. Place files in training_data/skin_disease/ folder
3. Run: python training_scripts/train_skin_model.py

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

HAM10000_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

def create_skin_model(num_classes=7):
    """Create ResNet50 model for skin disease classification"""
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_ham10000_data(data_dir):
    """Load and preprocess HAM10000 dataset"""
    
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return None, None
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(df)} entries")
    
    image_dirs = [
        os.path.join(data_dir, 'HAM10000_images_part_1'),
        os.path.join(data_dir, 'HAM10000_images_part_2'),
        os.path.join(data_dir, 'HAM10000_images'),
        data_dir
    ]
    
    images = []
    labels = []
    
    print("Loading images...")
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        
        if dx not in HAM10000_CLASSES:
            continue
        
        image_found = False
        for img_dir in image_dirs:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
                img_path = os.path.join(img_dir, f"{image_id}{ext}")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        images.append(img_array)
                        labels.append(HAM10000_CLASSES.index(dx))
                        image_found = True
                        break
                    except Exception as e:
                        continue
            if image_found:
                break
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{len(df)} images...")
    
    if len(images) == 0:
        print("ERROR: No images were loaded!")
        return None, None
    
    X = np.array(images)
    y = to_categorical(labels, num_classes=len(HAM10000_CLASSES))
    
    print(f"Loaded {len(X)} images")
    return X, y

def train_skin_model():
    print("=" * 60)
    print("TRAINING SKIN DISEASE DETECTION MODEL")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_dir, 'training_data', 'skin_disease')
    weights_dir = os.path.join(project_dir, 'models', 'weights')
    
    os.makedirs(weights_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print(f"\nERROR: Training data not found!")
        print(f"Expected folder: {data_dir}")
        print("\nPlease download the HAM10000 dataset from Kaggle and place it in:")
        print("training_data/skin_disease/")
        print("\nFolder structure should be:")
        print("training_data/skin_disease/HAM10000_metadata.csv")
        print("training_data/skin_disease/HAM10000_images_part_1/")
        print("training_data/skin_disease/HAM10000_images_part_2/")
        return False
    
    print("\n[Step 1/3] Loading HAM10000 dataset...")
    X, y = load_ham10000_data(data_dir)
    
    if X is None:
        return False
    
    print(f"   Total images: {len(X)}")
    print(f"   Image shape: {X[0].shape}")
    print(f"   Classes: {len(HAM10000_CLASSES)}")
    
    print("\n[Step 2/3] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    print("\n[Step 3/3] Training ResNet50 model...")
    
    model = create_skin_model(num_classes=len(HAM10000_CLASSES))
    
    save_path = os.path.join(weights_dir, 'skin_resnet50.h5')
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    epochs = 20
    batch_size = 32
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    best_accuracy = max(history.history['val_accuracy'])
    print(f"\n   Best Accuracy: {best_accuracy:.2%}")
    print(f"   Saved: skin_resnet50.h5")
    
    print("\n" + "=" * 60)
    print("SKIN MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTrained model saved in: models/weights/skin_resnet50.h5")
    print("\nClasses trained:")
    for i, cls in enumerate(HAM10000_CLASSES):
        names = {
            'nv': 'Melanocytic Nevus (Mole)',
            'mel': 'Melanoma',
            'bkl': 'Benign Keratosis',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratosis',
            'vasc': 'Vascular Lesion',
            'df': 'Dermatofibroma'
        }
        print(f"   {i+1}. {names.get(cls, cls)}")
    print("\nRestart the app to use the trained model!")
    return True

if __name__ == "__main__":
    train_skin_model()

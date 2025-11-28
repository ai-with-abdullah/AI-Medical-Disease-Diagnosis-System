"""
Train Pneumonia Detection Models
================================
This script trains 3 CNN models for pneumonia detection from chest X-rays.

Models trained:
1. ResNet50 - Deep residual network
2. EfficientNetB0 - Efficient architecture
3. MobileNetV2 - Lightweight model

Usage:
------
1. First download the Kaggle Chest X-Ray dataset
2. Place images in training_data/pneumonia/ folder
3. Run: python training_scripts/train_pneumonia_models.py

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
"""

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(base_model_name, num_classes=2):
    """Create a CNN model with transfer learning"""
    
    if base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_pneumonia_models():
    print("=" * 60)
    print("TRAINING PNEUMONIA DETECTION MODELS")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    train_dir = os.path.join(project_dir, 'training_data', 'pneumonia', 'train')
    val_dir = os.path.join(project_dir, 'training_data', 'pneumonia', 'val')
    weights_dir = os.path.join(project_dir, 'models', 'weights')
    
    os.makedirs(weights_dir, exist_ok=True)
    
    if not os.path.exists(train_dir):
        print(f"\nERROR: Training data not found!")
        print(f"Expected folder: {train_dir}")
        print("\nPlease download the Kaggle Chest X-Ray dataset and place it in:")
        print("training_data/pneumonia/")
        print("\nFolder structure should be:")
        print("training_data/pneumonia/train/NORMAL/")
        print("training_data/pneumonia/train/PNEUMONIA/")
        print("training_data/pneumonia/val/NORMAL/")
        print("training_data/pneumonia/val/PNEUMONIA/")
        return False
    
    print("\n[Step 1/4] Setting up data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    batch_size = 32
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['NORMAL', 'PNEUMONIA']
    )
    
    if os.path.exists(val_dir):
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            classes=['NORMAL', 'PNEUMONIA']
        )
    else:
        print("WARNING: Validation folder not found, using training data for validation")
        val_generator = train_generator
    
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    
    models_to_train = [
        ('resnet50', 'pneumonia_resnet50.h5'),
        ('efficientnet', 'pneumonia_efficientnet.h5'),
        ('mobilenet', 'pneumonia_mobilenet.h5')
    ]
    
    epochs = 15
    
    for idx, (model_name, save_name) in enumerate(models_to_train, 1):
        print(f"\n[Step {idx+1}/4] Training {model_name.upper()} model...")
        
        model = create_model(model_name, num_classes=2)
        
        save_path = os.path.join(weights_dir, save_name)
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=3,
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
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        best_accuracy = max(history.history['val_accuracy'])
        print(f"   Best Accuracy: {best_accuracy:.2%}")
        print(f"   Saved: {save_name}")
    
    print("\n" + "=" * 60)
    print("PNEUMONIA MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTrained models saved in: models/weights/")
    print("- pneumonia_resnet50.h5")
    print("- pneumonia_efficientnet.h5")
    print("- pneumonia_mobilenet.h5")
    print("\nRestart the app to use trained models!")
    return True

if __name__ == "__main__":
    train_pneumonia_models()

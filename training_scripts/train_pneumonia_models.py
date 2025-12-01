"""
Train Pneumonia Detection Models
================================
This script trains 3 CNN models for pneumonia detection from chest X-rays.
It automatically detects and combines multiple datasets for higher accuracy.

Models trained:
1. ResNet50 - Deep residual network
2. EfficientNetB0 - Efficient architecture
3. MobileNetV2 - Lightweight model

Supported Datasets (8 total, auto-detected):
1. Kaggle Chest X-Ray (5,863 images) - folder: kaggle/ or train/
2. RSNA Pneumonia Detection (26,684 images) - folder: rsna/
3. NIH ChestX-ray14 (112,120 images) - folder: nih/
4. COVID-Pneumonia-Normal (5,228 images) - folder: covid_pneumonia_normal/
5. Roboflow Chest X-Rays (3,000+ images) - folder: roboflow/
6. VinDr-CXR (18,000 images) - folder: vindr/
7. CheXpert Stanford (224,316 images) - folder: chexpert/
8. NIH Resized 224x224 (112,120 images) - folder: nih_resized/ or nih_224x224/

Note: Each dataset can be placed in organized format (NORMAL/PNEUMONIA subfolders)
      or raw format with CSV labels. Organized format is preferred for reliability.

Usage:
------
1. Download datasets and place in training_data/pneumonia/ folder
2. Run: python training_scripts/train_pneumonia_models.py

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
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

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
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def detect_datasets(pneumonia_dir):
    """Detect which datasets are available"""
    datasets_found = []
    total_images = 0

    print("\n[Detecting Available Datasets...]")
    print("-" * 50)

    # Dataset 1: Kaggle Chest X-Ray
    kaggle_paths = [
        os.path.join(pneumonia_dir, 'kaggle', 'train'),
        os.path.join(pneumonia_dir, 'train'),  # Direct placement
    ]
    for kaggle_train in kaggle_paths:
        if os.path.exists(kaggle_train):
            normal_count = len(os.listdir(os.path.join(kaggle_train, 'NORMAL'))) if os.path.exists(os.path.join(kaggle_train, 'NORMAL')) else 0
            pneumonia_count = len(os.listdir(os.path.join(kaggle_train, 'PNEUMONIA'))) if os.path.exists(os.path.join(kaggle_train, 'PNEUMONIA')) else 0
            if normal_count > 0 or pneumonia_count > 0:
                datasets_found.append({
                    'name': 'Kaggle Chest X-Ray',
                    'path': kaggle_train,
                    'type': 'directory',
                    'normal': normal_count,
                    'pneumonia': pneumonia_count
                })
                total_images += normal_count + pneumonia_count
                print(f"   [+] Kaggle Chest X-Ray: {normal_count + pneumonia_count} images")
                break

    # Dataset 2: RSNA (converted to PNG)
    rsna_paths = [
        os.path.join(pneumonia_dir, 'rsna', 'images_png'),
        os.path.join(pneumonia_dir, 'rsna', 'train'),
    ]
    for rsna_dir in rsna_paths:
        if os.path.exists(rsna_dir):
            img_count = len([f for f in os.listdir(rsna_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            if img_count > 0:
                label_file = os.path.join(pneumonia_dir, 'rsna', 'stage_2_train_labels.csv')
                datasets_found.append({
                    'name': 'RSNA Pneumonia',
                    'path': rsna_dir,
                    'type': 'rsna',
                    'label_file': label_file if os.path.exists(label_file) else None,
                    'count': img_count
                })
                total_images += img_count
                print(f"   [+] RSNA Pneumonia: {img_count} images")
                break

    # Dataset 3: NIH ChestX-ray14
    nih_paths = [
        os.path.join(pneumonia_dir, 'nih', 'organized'),
        os.path.join(pneumonia_dir, 'nih', 'images'),
    ]
    for nih_dir in nih_paths:
        if os.path.exists(nih_dir):
            if os.path.exists(os.path.join(nih_dir, 'NORMAL')) and os.path.exists(os.path.join(nih_dir, 'PNEUMONIA')):
                normal_count = len(os.listdir(os.path.join(nih_dir, 'NORMAL')))
                pneumonia_count = len(os.listdir(os.path.join(nih_dir, 'PNEUMONIA')))
                datasets_found.append({
                    'name': 'NIH ChestX-ray14',
                    'path': nih_dir,
                    'type': 'directory',
                    'normal': normal_count,
                    'pneumonia': pneumonia_count
                })
                total_images += normal_count + pneumonia_count
                print(f"   [+] NIH ChestX-ray14: {normal_count + pneumonia_count} images")
                break
            else:
                label_file = os.path.join(pneumonia_dir, 'nih', 'Data_Entry_2017.csv')
                if os.path.exists(label_file):
                    img_count = len([f for f in os.listdir(nih_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    if img_count > 0:
                        datasets_found.append({
                            'name': 'NIH ChestX-ray14 (raw)',
                            'path': nih_dir,
                            'type': 'nih_raw',
                            'label_file': label_file,
                            'count': img_count
                        })
                        total_images += img_count
                        print(f"   [+] NIH ChestX-ray14 (raw): {img_count} images (needs preprocessing)")
                        break

    # Dataset 4: COVID-Pneumonia-Normal
    covid_dir = os.path.join(pneumonia_dir, 'covid_pneumonia_normal')
    if os.path.exists(covid_dir):
        normal_count = len(os.listdir(os.path.join(covid_dir, 'NORMAL'))) if os.path.exists(os.path.join(covid_dir, 'NORMAL')) else 0
        pneumonia_count = len(os.listdir(os.path.join(covid_dir, 'PNEUMONIA'))) if os.path.exists(os.path.join(covid_dir, 'PNEUMONIA')) else 0
        covid_count = len(os.listdir(os.path.join(covid_dir, 'COVID'))) if os.path.exists(os.path.join(covid_dir, 'COVID')) else 0
        if normal_count > 0 or pneumonia_count > 0:
            datasets_found.append({
                'name': 'COVID-Pneumonia-Normal',
                'path': covid_dir,
                'type': 'covid',
                'normal': normal_count,
                'pneumonia': pneumonia_count,
                'covid': covid_count
            })
            total_images += normal_count + pneumonia_count + covid_count
            print(f"   [+] COVID-Pneumonia-Normal: {normal_count + pneumonia_count + covid_count} images")

    # Dataset 5: Roboflow
    roboflow_paths = [
        os.path.join(pneumonia_dir, 'roboflow', 'train'),
        os.path.join(pneumonia_dir, 'roboflow'),
    ]
    for roboflow_dir in roboflow_paths:
        if os.path.exists(roboflow_dir):
            normal_path = os.path.join(roboflow_dir, 'NORMAL')
            pneumonia_path = os.path.join(roboflow_dir, 'PNEUMONIA')
            if os.path.exists(normal_path) or os.path.exists(pneumonia_path):
                normal_count = len(os.listdir(normal_path)) if os.path.exists(normal_path) else 0
                pneumonia_count = len(os.listdir(pneumonia_path)) if os.path.exists(pneumonia_path) else 0
                if normal_count > 0 or pneumonia_count > 0:
                    datasets_found.append({
                        'name': 'Roboflow',
                        'path': roboflow_dir,
                        'type': 'directory',
                        'normal': normal_count,
                        'pneumonia': pneumonia_count
                    })
                    total_images += normal_count + pneumonia_count
                    print(f"   [+] Roboflow: {normal_count + pneumonia_count} images")
                    break

    # Dataset 6: VinDr-CXR
    vindr_dir = os.path.join(pneumonia_dir, 'vindr')
    if os.path.exists(vindr_dir):
        if os.path.exists(os.path.join(vindr_dir, 'NORMAL')) and os.path.exists(os.path.join(vindr_dir, 'PNEUMONIA')):
            normal_count = len(os.listdir(os.path.join(vindr_dir, 'NORMAL')))
            pneumonia_count = len(os.listdir(os.path.join(vindr_dir, 'PNEUMONIA')))
            datasets_found.append({
                'name': 'VinDr-CXR',
                'path': vindr_dir,
                'type': 'directory',
                'normal': normal_count,
                'pneumonia': pneumonia_count
            })
            total_images += normal_count + pneumonia_count
            print(f"   [+] VinDr-CXR: {normal_count + pneumonia_count} images")

    # Dataset 7: CheXpert (Stanford)
    chexpert_paths = [
        os.path.join(pneumonia_dir, 'chexpert', 'train'),
        os.path.join(pneumonia_dir, 'chexpert'),
    ]
    for chexpert_dir in chexpert_paths:
        if os.path.exists(chexpert_dir):
            # Check for organized structure first
            if os.path.exists(os.path.join(chexpert_dir, 'NORMAL')) and os.path.exists(os.path.join(chexpert_dir, 'PNEUMONIA')):
                normal_count = len(os.listdir(os.path.join(chexpert_dir, 'NORMAL')))
                pneumonia_count = len(os.listdir(os.path.join(chexpert_dir, 'PNEUMONIA')))
                datasets_found.append({
                    'name': 'CheXpert',
                    'path': chexpert_dir,
                    'type': 'directory',
                    'normal': normal_count,
                    'pneumonia': pneumonia_count
                })
                total_images += normal_count + pneumonia_count
                print(f"   [+] CheXpert: {normal_count + pneumonia_count} images")
                break
            # Check for raw CheXpert structure with CSV
            train_csv = os.path.join(pneumonia_dir, 'chexpert', 'train.csv')
            if os.path.exists(train_csv):
                # Count patient folders
                patient_count = 0
                for item in os.listdir(chexpert_dir):
                    if os.path.isdir(os.path.join(chexpert_dir, item)) and item.startswith('patient'):
                        patient_count += 1
                if patient_count > 0:
                    datasets_found.append({
                        'name': 'CheXpert (raw)',
                        'path': chexpert_dir,
                        'type': 'chexpert_raw',
                        'label_file': train_csv,
                        'count': patient_count
                    })
                    total_images += patient_count * 2  # Estimate 2 images per patient
                    print(f"   [+] CheXpert (raw): ~{patient_count * 2} images (needs preprocessing)")
                    break

    # Dataset 8: NIH Resized (224x224) - Pre-processed version
    nih_resized_paths = [
        os.path.join(pneumonia_dir, 'nih_resized'),
        os.path.join(pneumonia_dir, 'nih_224x224'),
        os.path.join(pneumonia_dir, 'nih-chest-x-ray-14-224x224-resized'),
    ]
    for nih_resized_dir in nih_resized_paths:
        if os.path.exists(nih_resized_dir):
            # Check for organized structure
            if os.path.exists(os.path.join(nih_resized_dir, 'NORMAL')) and os.path.exists(os.path.join(nih_resized_dir, 'PNEUMONIA')):
                normal_count = len(os.listdir(os.path.join(nih_resized_dir, 'NORMAL')))
                pneumonia_count = len(os.listdir(os.path.join(nih_resized_dir, 'PNEUMONIA')))
                datasets_found.append({
                    'name': 'NIH Resized 224x224',
                    'path': nih_resized_dir,
                    'type': 'directory',
                    'normal': normal_count,
                    'pneumonia': pneumonia_count
                })
                total_images += normal_count + pneumonia_count
                print(f"   [+] NIH Resized 224x224: {normal_count + pneumonia_count} images")
                break
            # Check for images folder with CSV
            images_dir = os.path.join(nih_resized_dir, 'images')
            if os.path.exists(images_dir):
                img_count = len([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 0:
                    label_file = None
                    for csv_name in ['Data_Entry_2017.csv', 'labels.csv', 'data_entry.csv']:
                        csv_path = os.path.join(nih_resized_dir, csv_name)
                        if os.path.exists(csv_path):
                            label_file = csv_path
                            break
                    datasets_found.append({
                        'name': 'NIH Resized 224x224 (raw)',
                        'path': images_dir,
                        'type': 'nih_raw',
                        'label_file': label_file,
                        'count': img_count
                    })
                    total_images += img_count
                    print(f"   [+] NIH Resized 224x224 (raw): {img_count} images (needs preprocessing)")
                    break
            # Direct images in folder
            else:
                img_count = len([f for f in os.listdir(nih_resized_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 1000:  # Likely the NIH resized dataset
                    label_file = None
                    for csv_name in ['Data_Entry_2017.csv', 'labels.csv', 'data_entry.csv']:
                        csv_path = os.path.join(nih_resized_dir, csv_name)
                        if os.path.exists(csv_path):
                            label_file = csv_path
                            break
                    datasets_found.append({
                        'name': 'NIH Resized 224x224 (raw)',
                        'path': nih_resized_dir,
                        'type': 'nih_raw',
                        'label_file': label_file,
                        'count': img_count
                    })
                    total_images += img_count
                    print(f"   [+] NIH Resized 224x224 (raw): {img_count} images (needs preprocessing)")
                    break

    print("-" * 50)
    print(f"   TOTAL: {len(datasets_found)} datasets, {total_images} images")

    return datasets_found, total_images


def combine_datasets(datasets, output_dir):
    """Combine multiple datasets into a single training directory"""

    print("\n[Combining Datasets...]")

    train_dir = os.path.join(output_dir, 'combined_train')
    val_dir = os.path.join(output_dir, 'combined_val')

    # Create directories
    for dir_path in [train_dir, val_dir]:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

    normal_count = 0
    pneumonia_count = 0

    for dataset in datasets:
        print(f"   Processing: {dataset['name']}")

        if dataset['type'] == 'directory':
            # Standard directory structure with NORMAL and PNEUMONIA folders
            normal_src = os.path.join(dataset['path'], 'NORMAL')
            pneumonia_src = os.path.join(dataset['path'], 'PNEUMONIA')

            if os.path.exists(normal_src):
                for img in os.listdir(normal_src):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src = os.path.join(normal_src, img)
                        dst = os.path.join(train_dir, 'NORMAL', f"{dataset['name'].replace(' ', '_')}_{img}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            normal_count += 1

            if os.path.exists(pneumonia_src):
                for img in os.listdir(pneumonia_src):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src = os.path.join(pneumonia_src, img)
                        dst = os.path.join(train_dir, 'PNEUMONIA', f"{dataset['name'].replace(' ', '_')}_{img}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            pneumonia_count += 1

        elif dataset['type'] == 'covid':
            # COVID-Pneumonia-Normal dataset (treat COVID as PNEUMONIA for binary classification)
            normal_src = os.path.join(dataset['path'], 'NORMAL')
            pneumonia_src = os.path.join(dataset['path'], 'PNEUMONIA')
            covid_src = os.path.join(dataset['path'], 'COVID')

            if os.path.exists(normal_src):
                for img in os.listdir(normal_src):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src = os.path.join(normal_src, img)
                        dst = os.path.join(train_dir, 'NORMAL', f"covid_dataset_{img}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            normal_count += 1

            # Combine PNEUMONIA and COVID as PNEUMONIA
            for src_dir in [pneumonia_src, covid_src]:
                if os.path.exists(src_dir):
                    for img in os.listdir(src_dir):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src = os.path.join(src_dir, img)
                            dst = os.path.join(train_dir, 'PNEUMONIA', f"covid_dataset_{img}")
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
                                pneumonia_count += 1

        elif dataset['type'] == 'rsna' and dataset.get('label_file'):
            # RSNA dataset with labels CSV
            try:
                labels_df = pd.read_csv(dataset['label_file'])
                for _, row in labels_df.iterrows():
                    patient_id = row['patientId']
                    target = row['Target'] if 'Target' in row else 0

                    # Find image file
                    for ext in ['.png', '.jpg', '.jpeg']:
                        img_path = os.path.join(dataset['path'], f"{patient_id}{ext}")
                        if os.path.exists(img_path):
                            class_name = 'PNEUMONIA' if target == 1 else 'NORMAL'
                            dst = os.path.join(train_dir, class_name, f"rsna_{patient_id}{ext}")
                            if not os.path.exists(dst):
                                shutil.copy2(img_path, dst)
                                if target == 1:
                                    pneumonia_count += 1
                                else:
                                    normal_count += 1
                            break
            except Exception as e:
                print(f"   Warning: Could not process RSNA labels: {e}")

        elif dataset['type'] == 'nih_raw' and dataset.get('label_file'):
            # NIH dataset with labels CSV - extract pneumonia and normal cases
            try:
                labels_df = pd.read_csv(dataset['label_file'])

                # Filter pneumonia cases
                pneumonia_df = labels_df[labels_df['Finding Labels'].str.contains('Pneumonia', na=False)]
                normal_df = labels_df[labels_df['Finding Labels'] == 'No Finding']

                # Balance classes - take same number of normal as pneumonia
                if len(normal_df) > len(pneumonia_df):
                    normal_df = normal_df.sample(n=min(len(pneumonia_df), len(normal_df)), random_state=42)

                # Copy pneumonia images
                for _, row in pneumonia_df.iterrows():
                    img_name = row['Image Index']
                    img_path = os.path.join(dataset['path'], img_name)
                    if os.path.exists(img_path):
                        dst = os.path.join(train_dir, 'PNEUMONIA', f"nih_{img_name}")
                        if not os.path.exists(dst):
                            shutil.copy2(img_path, dst)
                            pneumonia_count += 1

                # Copy normal images
                for _, row in normal_df.iterrows():
                    img_name = row['Image Index']
                    img_path = os.path.join(dataset['path'], img_name)
                    if os.path.exists(img_path):
                        dst = os.path.join(train_dir, 'NORMAL', f"nih_{img_name}")
                        if not os.path.exists(dst):
                            shutil.copy2(img_path, dst)
                            normal_count += 1

                print(f"      Extracted {pneumonia_count} pneumonia, {normal_count} normal from NIH")
            except Exception as e:
                print(f"   Warning: Could not process NIH labels: {e}")

        elif dataset['type'] == 'chexpert_raw' and dataset.get('label_file'):
            # CheXpert dataset with labels CSV
            try:
                labels_df = pd.read_csv(dataset['label_file'])

                # CheXpert uses different column names - check for consolidation/pneumonia related columns
                # Columns include: Consolidation, Lung Opacity, Pneumonia (sometimes)
                pneumonia_cols = ['Pneumonia', 'Consolidation', 'Lung Opacity']
                available_cols = [col for col in pneumonia_cols if col in labels_df.columns]

                if available_cols:
                    # Filter positive cases (1.0 = positive)
                    pneumonia_mask = labels_df[available_cols].apply(lambda x: x == 1.0).any(axis=1)
                    pneumonia_df = labels_df[pneumonia_mask]

                    # Filter normal cases (all conditions are 0.0 or -1.0)
                    all_condition_cols = [col for col in labels_df.columns if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
                    normal_mask = labels_df[all_condition_cols].apply(lambda x: (x == 0.0) | (x == -1.0)).all(axis=1)
                    normal_df = labels_df[normal_mask]

                    # Balance classes
                    if len(normal_df) > len(pneumonia_df):
                        normal_df = normal_df.sample(n=min(len(pneumonia_df), len(normal_df)), random_state=42)

                    # Copy images
                    base_dir = os.path.dirname(dataset['label_file'])

                    for _, row in pneumonia_df.iterrows():
                        img_path = os.path.join(base_dir, row['Path'])
                        if os.path.exists(img_path):
                            img_name = os.path.basename(img_path)
                            dst = os.path.join(train_dir, 'PNEUMONIA', f"chexpert_{img_name}")
                            if not os.path.exists(dst):
                                shutil.copy2(img_path, dst)
                                pneumonia_count += 1

                    for _, row in normal_df.iterrows():
                        img_path = os.path.join(base_dir, row['Path'])
                        if os.path.exists(img_path):
                            img_name = os.path.basename(img_path)
                            dst = os.path.join(train_dir, 'NORMAL', f"chexpert_{img_name}")
                            if not os.path.exists(dst):
                                shutil.copy2(img_path, dst)
                                normal_count += 1

                    print(f"      Extracted {pneumonia_count} pneumonia, {normal_count} normal from CheXpert")
            except Exception as e:
                print(f"   Warning: Could not process CheXpert labels: {e}")

    # Create validation split (10% of training data)
    print("\n   Creating validation split (10%)...")
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        images = os.listdir(class_dir)
        np.random.shuffle(images)
        val_count = int(len(images) * 0.1)

        for img in images[:val_count]:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.move(src, dst)

    final_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    final_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    val_normal = len(os.listdir(os.path.join(val_dir, 'NORMAL')))
    val_pneumonia = len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))

    print(f"\n   Combined Dataset Summary:")
    print(f"   - Training: {final_normal} Normal, {final_pneumonia} Pneumonia")
    print(f"   - Validation: {val_normal} Normal, {val_pneumonia} Pneumonia")
    print(f"   - Total: {final_normal + final_pneumonia + val_normal + val_pneumonia} images")

    return train_dir, val_dir


def train_pneumonia_models():
    print("=" * 60)
    print("TRAINING PNEUMONIA DETECTION MODELS")
    print("Multi-Dataset Support with Auto-Detection")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    pneumonia_dir = os.path.join(project_dir, 'training_data', 'pneumonia')
    weights_dir = os.path.join(project_dir, 'models', 'weights')

    os.makedirs(weights_dir, exist_ok=True)

    # Detect available datasets
    datasets, total_images = detect_datasets(pneumonia_dir)

    if len(datasets) == 0:
        print(f"\nERROR: No training data found!")
        print(f"Expected folder: {pneumonia_dir}")
        print("\nPlease download at least one of these datasets:")
        print("1. Kaggle Chest X-Ray: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. RSNA Pneumonia: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data")
        print("3. NIH ChestX-ray14: https://www.kaggle.com/datasets/nih-chest-xrays/data")
        print("4. COVID-Pneumonia-Normal: https://data.mendeley.com/datasets/dvntn9yhd2/1")
        print("\nSee COMPREHENSIVE_TRAINING_GUIDE.md for detailed instructions.")
        return False

    # Check if we should combine datasets or use single dataset
    if len(datasets) > 1:
        print(f"\n[Multiple datasets detected - combining for better accuracy...]")
        train_dir, val_dir = combine_datasets(datasets, pneumonia_dir)
    else:
        # Single dataset - use it directly
        dataset = datasets[0]
        if dataset['type'] == 'directory':
            train_dir = dataset['path']
            # Check for validation folder
            parent_dir = os.path.dirname(train_dir)
            val_dir = os.path.join(parent_dir, 'val')
            if not os.path.exists(val_dir):
                val_dir = train_dir  # Use training data for validation if no val folder
        else:
            # Need to organize the dataset first
            print("\n[Organizing single dataset...]")
            train_dir, val_dir = combine_datasets(datasets, pneumonia_dir)

    print("\n[Step 1/4] Setting up data generators...")

    # Enhanced data augmentation for better accuracy
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Adjust batch size based on dataset size
    if total_images > 50000:
        batch_size = 64
    elif total_images > 10000:
        batch_size = 32
    else:
        batch_size = 16

    print(f"   Batch size: {batch_size}")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['NORMAL', 'PNEUMONIA'],
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['NORMAL', 'PNEUMONIA'],
        shuffle=False
    )

    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")

    # Calculate class weights for imbalanced data
    class_counts = [0, 0]
    class_counts[0] = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    class_counts[1] = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    total = sum(class_counts)
    class_weights = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
    print(f"   Class weights: Normal={class_weights[0]:.2f}, Pneumonia={class_weights[1]:.2f}")

    models_to_train = [
        ('resnet50', 'pneumonia_resnet50.h5'),
        ('efficientnet', 'pneumonia_efficientnet.h5'),
        ('mobilenet', 'pneumonia_mobilenet.h5')
    ]

    # Adjust epochs based on dataset size
    if total_images > 50000:
        epochs = 20
        fine_tune_epochs = 10
    elif total_images > 10000:
        epochs = 15
        fine_tune_epochs = 5
    else:
        epochs = 10
        fine_tune_epochs = 3

    print(f"\n   Training epochs: {epochs} (+ {fine_tune_epochs} fine-tuning)")

    for idx, (model_name, save_name) in enumerate(models_to_train, 1):
        print(f"\n[Step {idx+1}/4] Training {model_name.upper()} model...")

        model, base_model = create_model(model_name, num_classes=2)

        save_path = os.path.join(weights_dir, save_name)

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
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Phase 1: Train with frozen base model
        print(f"   Phase 1: Training with frozen base model...")
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        best_accuracy = max(history.history['val_accuracy'])
        print(f"   Phase 1 Best Accuracy: {best_accuracy:.2%}")

        # Phase 2: Fine-tune last layers of base model
        if best_accuracy < 0.95 and fine_tune_epochs > 0:
            print(f"   Phase 2: Fine-tuning last 20 layers...")

            # Unfreeze last 20 layers
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False

            # Recompile with lower learning rate
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            history_fine = model.fit(
                train_generator,
                epochs=fine_tune_epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            best_accuracy = max(max(history.history['val_accuracy']), max(history_fine.history['val_accuracy']))

        print(f"   Final Best Accuracy: {best_accuracy:.2%}")
        print(f"   Saved: {save_name}")

    print("\n" + "=" * 60)
    print("PNEUMONIA MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTrained models saved in: models/weights/")
    print("- pneumonia_resnet50.h5")
    print("- pneumonia_efficientnet.h5")
    print("- pneumonia_mobilenet.h5")
    print("\nRestart the app to use trained models!")

    # Cleanup combined directory if created
    combined_dir = os.path.join(pneumonia_dir, 'combined_train')
    if os.path.exists(combined_dir):
        print(f"\nNote: Combined training data saved in {combined_dir}")
        print("You can delete this folder to save disk space after training.")

    return True


if __name__ == "__main__":
    train_pneumonia_models()

"""
Train Skin Cancer Model - ALL-IN-ONE SCRIPT (Multi-Dataset Support)
=====================================================================
Just download dataset(s) and run this script - no code changes needed!

This script automatically:
1. Detects ALL skin cancer datasets in training_data/skin_cancer/
2. Combines multiple datasets for better accuracy
3. Organizes images into class folders if needed
4. Trains ResNet50 model with transfer learning
5. Saves trained model to models/weights/

Supported Datasets (place in training_data/skin_cancer/):
----------------------------------------------------------
1. HAM10000 (Recommended)
   - Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Size: 2.7 GB, 10,015 images, 7 classes

2. ISIC 2019 Challenge (High Accuracy)
   - Link: https://www.kaggle.com/datasets/andrewmvd/isic-2019
   - Size: 9 GB, 25,331 images, 8 classes

3. ISIC 2020 Challenge (Melanoma Focus)
   - Link: https://www.kaggle.com/competitions/siim-isic-melanoma-classification
   - Size: 15 GB, 33,126 images, Binary classification

4. PAD-UFES-20 (Smartphone Images)
   - Link: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
   - Size: 500 MB, 2,298 images, 6 classes

5. Melanoma Binary Dataset
   - Link: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
   - Size: 3 GB, 10,605 images, Binary

6. Any pre-organized dataset with class folders

Usage:
------
1. Download any combination of datasets
2. Extract to training_data/skin_cancer/
3. Run: python training_scripts/train_skin_model.py

That's it! The script auto-detects and combines all available data.

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
    'MEL': 'mel',
    'NV': 'nv',
    'BCC': 'bcc',
    'AK': 'akiec',
    'BKL': 'bkl',
    'DF': 'df',
    'VASC': 'vasc',
    'SCC': 'bcc',
    'UNK': None
}

PAD_UFES_TO_HAM10000 = {
    'ACK': 'akiec',
    'BCC': 'bcc',
    'MEL': 'mel',
    'NEV': 'nv',
    'SCC': 'bcc',
    'SEK': 'bkl'
}

BINARY_TO_HAM10000 = {
    'benign': 'nv',
    'malignant': 'mel',
    'Benign': 'nv',
    'Malignant': 'mel'
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


def get_data_directory():
    """Get the skin cancer data directory (check both old and new names)"""
    if os.path.exists(SKIN_DATA_DIR):
        return SKIN_DATA_DIR
    elif os.path.exists(SKIN_DATA_DIR_OLD):
        print(f"   Note: Using legacy folder 'skin_disease'. Consider renaming to 'skin_cancer'.")
        return SKIN_DATA_DIR_OLD
    else:
        os.makedirs(SKIN_DATA_DIR, exist_ok=True)
        return SKIN_DATA_DIR


def detect_all_datasets():
    """Detect all available skin cancer datasets"""
    print("\n" + "=" * 70)
    print("STEP 1: DETECTING SKIN CANCER DATASETS")
    print("=" * 70)
    
    data_dir = get_data_directory()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    datasets = {
        'ham10000': None,
        'isic2019': None,
        'isic2020': None,
        'pad_ufes_20': None,
        'melanoma_binary': None,
        'organized': None
    }
    
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
    
    isic2019_dirs = [
        os.path.join(data_dir, 'isic2019'),
        os.path.join(data_dir, 'ISIC2019'),
        os.path.join(data_dir, 'isic_2019')
    ]
    
    for isic_dir in isic2019_dirs:
        if os.path.exists(isic_dir):
            gt_files = [
                os.path.join(isic_dir, 'ISIC_2019_Training_GroundTruth.csv'),
                os.path.join(isic_dir, 'ground_truth.csv'),
                os.path.join(isic_dir, 'labels.csv')
            ]
            
            img_dirs = [
                os.path.join(isic_dir, 'ISIC_2019_Training_Input'),
                os.path.join(isic_dir, 'images'),
                isic_dir
            ]
            
            gt_file = None
            for f in gt_files:
                if os.path.exists(f):
                    gt_file = f
                    break
            
            img_dir = None
            for d in img_dirs:
                if os.path.exists(d) and os.path.isdir(d):
                    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(imgs) > 100:
                        img_dir = d
                        break
            
            if gt_file and img_dir:
                img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"\n   [FOUND] ISIC 2019 dataset: {img_count:,} images")
                datasets['isic2019'] = {
                    'type': 'isic2019',
                    'metadata': gt_file,
                    'image_dir': img_dir,
                    'count': img_count
                }
                total_images += img_count
            break
    
    isic2020_dirs = [
        os.path.join(data_dir, 'isic2020'),
        os.path.join(data_dir, 'ISIC2020'),
        os.path.join(data_dir, 'siim-isic-melanoma-classification')
    ]
    
    for isic_dir in isic2020_dirs:
        if os.path.exists(isic_dir):
            csv_files = [
                os.path.join(isic_dir, 'train.csv'),
                os.path.join(isic_dir, 'ISIC_2020_Training_GroundTruth.csv')
            ]
            
            img_dirs = [
                os.path.join(isic_dir, 'train'),
                os.path.join(isic_dir, 'jpeg', 'train'),
                os.path.join(isic_dir, 'images')
            ]
            
            csv_file = None
            for f in csv_files:
                if os.path.exists(f):
                    csv_file = f
                    break
            
            img_dir = None
            for d in img_dirs:
                if os.path.exists(d) and os.path.isdir(d):
                    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(imgs) > 100:
                        img_dir = d
                        break
            
            if csv_file and img_dir:
                img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"\n   [FOUND] ISIC 2020 dataset: {img_count:,} images")
                datasets['isic2020'] = {
                    'type': 'isic2020',
                    'metadata': csv_file,
                    'image_dir': img_dir,
                    'count': img_count
                }
                total_images += img_count
            break
    
    pad_dirs = [
        os.path.join(data_dir, 'pad_ufes_20'),
        os.path.join(data_dir, 'PAD-UFES-20'),
        os.path.join(data_dir, 'pad-ufes-20')
    ]
    
    for pad_dir in pad_dirs:
        if os.path.exists(pad_dir):
            csv_files = [
                os.path.join(pad_dir, 'metadata.csv'),
                os.path.join(pad_dir, 'PAD-UFES-20_metadata.csv')
            ]
            
            img_dirs = [
                os.path.join(pad_dir, 'images'),
                pad_dir
            ]
            
            csv_file = None
            for f in csv_files:
                if os.path.exists(f):
                    csv_file = f
                    break
            
            img_dir = None
            for d in img_dirs:
                if os.path.exists(d) and os.path.isdir(d):
                    imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(imgs) > 50:
                        img_dir = d
                        break
            
            if csv_file and img_dir:
                img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"\n   [FOUND] PAD-UFES-20 dataset: {img_count:,} images")
                datasets['pad_ufes_20'] = {
                    'type': 'pad_ufes_20',
                    'metadata': csv_file,
                    'image_dir': img_dir,
                    'count': img_count
                }
                total_images += img_count
            break
    
    binary_dirs = [
        os.path.join(data_dir, 'melanoma_binary'),
        os.path.join(data_dir, 'melanoma-skin-cancer-dataset'),
        os.path.join(data_dir, 'binary')
    ]
    
    for bin_dir in binary_dirs:
        if os.path.exists(bin_dir):
            benign_dir = None
            malignant_dir = None
            
            for subdir in os.listdir(bin_dir):
                subdir_lower = subdir.lower()
                subdir_path = os.path.join(bin_dir, subdir)
                if os.path.isdir(subdir_path):
                    if 'benign' in subdir_lower:
                        benign_dir = subdir_path
                    elif 'malignant' in subdir_lower or 'melanoma' in subdir_lower:
                        malignant_dir = subdir_path
            
            if benign_dir and malignant_dir:
                benign_count = len([f for f in os.listdir(benign_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                malignant_count = len([f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                img_count = benign_count + malignant_count
                
                print(f"\n   [FOUND] Melanoma Binary dataset: {img_count:,} images")
                print(f"      - Benign: {benign_count:,}")
                print(f"      - Malignant: {malignant_count:,}")
                
                datasets['melanoma_binary'] = {
                    'type': 'melanoma_binary',
                    'benign_dir': benign_dir,
                    'malignant_dir': malignant_dir,
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
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if folder in ['organized', 'ham10000', 'HAM10000', 'isic2019', 'isic2020', 
                      'pad_ufes_20', 'melanoma_binary', '__pycache__']:
            continue
        
        if os.path.isdir(folder_path):
            subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if len(subfolders) >= 2:
                has_images = False
                for sf in subfolders[:3]:
                    sf_path = os.path.join(folder_path, sf)
                    images = [f for f in os.listdir(sf_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(images) > 10:
                        has_images = True
                        break
                
                if has_images and datasets['organized'] is None:
                    print(f"\n   [FOUND] Class-organized dataset in {folder}/")
                    img_count = 0
                    for cls in subfolders:
                        cls_dir = os.path.join(folder_path, cls)
                        if os.path.isdir(cls_dir):
                            count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            img_count += count
                            print(f"      - {cls}: {count:,} images")
                    
                    datasets['organized'] = {
                        'type': 'organized',
                        'path': folder_path,
                        'count': img_count
                    }
                    total_images += img_count
    
    active_datasets = {k: v for k, v in datasets.items() if v is not None}
    
    if len(active_datasets) == 0:
        print("\n" + "!" * 70)
        print("ERROR: No skin cancer dataset found!")
        print("!" * 70)
        print("\nPlease download at least one dataset:")
        print("\n  Option 1: HAM10000 (Recommended)")
        print("  Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("\n  Option 2: ISIC 2019 (High Accuracy)")
        print("  Link: https://www.kaggle.com/datasets/andrewmvd/isic-2019")
        print("\nExtract to training_data/skin_cancer/")
        return None, data_dir
    
    print(f"\n   === DATASETS SUMMARY ===")
    print(f"   Total datasets found: {len(active_datasets)}")
    print(f"   Total images available: {total_images:,}")
    
    return active_datasets, data_dir


def organize_all_datasets(datasets, data_dir):
    """Organize all datasets into unified class folders"""
    print("\n" + "=" * 70)
    print("STEP 2: ORGANIZING DATASETS INTO UNIFIED STRUCTURE")
    print("=" * 70)
    
    organized_dir = os.path.join(data_dir, 'organized')
    
    if 'organized' in datasets and datasets['organized'] is not None:
        existing_count = datasets['organized']['count']
        total_new = sum(d['count'] for k, d in datasets.items() if k != 'organized' and d is not None)
        
        if existing_count >= total_new * 0.9 and total_new == 0:
            print(f"\n   Dataset already organized! ({existing_count:,} images)")
            return organized_dir
    
    for cls in HAM10000_CLASSES.keys():
        cls_dir = os.path.join(organized_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
    
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
            
            if (idx + 1) % 2000 == 0:
                print(f"      Progress: {idx + 1:,}/{len(df):,}")
        
        print(f"      Copied: {copied:,} images")
        total_copied += copied
    
    if datasets.get('isic2019'):
        print("\n   Processing ISIC 2019...")
        isic_data = datasets['isic2019']
        df = pd.read_csv(isic_data['metadata'])
        
        label_cols = [col for col in df.columns if col in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']]
        
        copied = 0
        for idx, row in df.iterrows():
            image_id = row.get('image', row.get('image_name', row.iloc[0]))
            
            pred_class = None
            for col in label_cols:
                if row.get(col, 0) == 1:
                    pred_class = col
                    break
            
            if pred_class is None:
                continue
            
            ham_class = ISIC2019_TO_HAM10000.get(pred_class)
            if ham_class is None:
                continue
            
            for ext in ['', '.jpg', '.jpeg', '.png']:
                src_path = os.path.join(isic_data['image_dir'], image_id + ext)
                if os.path.exists(src_path):
                    dst_file = os.path.join(organized_dir, ham_class, os.path.basename(src_path))
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_path, dst_file)
                        copied += 1
                    break
            
            if (idx + 1) % 5000 == 0:
                print(f"      Progress: {idx + 1:,}/{len(df):,}")
        
        print(f"      Copied: {copied:,} images")
        total_copied += copied
    
    if datasets.get('isic2020'):
        print("\n   Processing ISIC 2020...")
        isic_data = datasets['isic2020']
        df = pd.read_csv(isic_data['metadata'])
        
        copied = 0
        for idx, row in df.iterrows():
            image_id = row.get('image_name', row.get('image', row.iloc[0]))
            target = row.get('target', row.get('benign_malignant', 0))
            
            if target == 1 or str(target).lower() == 'malignant':
                ham_class = 'mel'
            else:
                ham_class = 'nv'
            
            for ext in ['', '.jpg', '.jpeg', '.png']:
                src_path = os.path.join(isic_data['image_dir'], image_id + ext)
                if os.path.exists(src_path):
                    dst_file = os.path.join(organized_dir, ham_class, os.path.basename(src_path))
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_path, dst_file)
                        copied += 1
                    break
            
            if (idx + 1) % 5000 == 0:
                print(f"      Progress: {idx + 1:,}/{len(df):,}")
        
        print(f"      Copied: {copied:,} images")
        total_copied += copied
    
    if datasets.get('pad_ufes_20'):
        print("\n   Processing PAD-UFES-20...")
        pad_data = datasets['pad_ufes_20']
        df = pd.read_csv(pad_data['metadata'])
        
        copied = 0
        for idx, row in df.iterrows():
            image_id = row.get('img_id', row.get('image_id', row.iloc[0]))
            diag = row.get('diagnostic', row.get('diagnosis', ''))
            
            ham_class = PAD_UFES_TO_HAM10000.get(diag)
            if ham_class is None:
                continue
            
            for ext in ['', '.png', '.jpg', '.jpeg']:
                src_path = os.path.join(pad_data['image_dir'], str(image_id) + ext)
                if os.path.exists(src_path):
                    dst_file = os.path.join(organized_dir, ham_class, os.path.basename(src_path))
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_path, dst_file)
                        copied += 1
                    break
            
            if (idx + 1) % 500 == 0:
                print(f"      Progress: {idx + 1:,}/{len(df):,}")
        
        print(f"      Copied: {copied:,} images")
        total_copied += copied
    
    if datasets.get('melanoma_binary'):
        print("\n   Processing Melanoma Binary...")
        bin_data = datasets['melanoma_binary']
        
        copied = 0
        for img_file in os.listdir(bin_data['benign_dir']):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(bin_data['benign_dir'], img_file)
                dst_file = os.path.join(organized_dir, 'nv', img_file)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_path, dst_file)
                    copied += 1
        
        for img_file in os.listdir(bin_data['malignant_dir']):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(bin_data['malignant_dir'], img_file)
                dst_file = os.path.join(organized_dir, 'mel', img_file)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_path, dst_file)
                    copied += 1
        
        print(f"      Copied: {copied:,} images")
        total_copied += copied
    
    print(f"\n   Organization complete!")
    print(f"   Total images organized: {total_copied:,}")
    
    print(f"\n   Class distribution:")
    for cls in HAM10000_CLASSES.keys():
        cls_dir = os.path.join(organized_dir, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            name = HAM10000_CLASSES[cls]
            print(f"      {cls} ({name}): {count:,} images")
    
    return organized_dir


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
    """Build ResNet50-based skin cancer classification model"""
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
    print("SKIN CANCER MODEL TRAINING - MULTI-DATASET SUPPORT")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Auto-detect ALL skin cancer datasets")
    print("  2. Combine multiple datasets for better accuracy")
    print("  3. Organize images by class")
    print("  4. Train ResNet50 model with transfer learning")
    print("  5. Save model to models/weights/skin_resnet50.h5")
    
    if not check_tensorflow():
        return False
    
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    datasets, data_dir = detect_all_datasets()
    if datasets is None:
        return False
    
    organized_dir = organize_all_datasets(datasets, data_dir)
    
    print("\n" + "=" * 70)
    print("STEP 3: PREPARING DATA GENERATORS")
    print("=" * 70)
    
    total_images = 0
    for cls in HAM10000_CLASSES.keys():
        cls_dir = os.path.join(organized_dir, cls)
        if os.path.exists(cls_dir):
            total_images += len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if total_images > 20000:
        batch_size = 64
    elif total_images > 5000:
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
    print("SUCCESS! Your skin cancer model is ready.")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    train_skin_model()

"""
Pneumonia Dataset Preprocessing Script
=======================================
This script handles all preprocessing tasks for pneumonia datasets:
1. RSNA DICOM to PNG conversion
2. NIH ChestX-ray14 pneumonia case extraction
3. VinDr-CXR DICOM conversion
4. CheXpert data organization

Run this script BEFORE training if you have raw datasets.
The training script will also auto-detect and run preprocessing if needed.

Usage:
------
python training_scripts/preprocess_pneumonia_data.py

Team Members:
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PNEUMONIA_DATA_DIR = str(PROJECT_ROOT / 'training_data' / 'pneumonia')


def convert_dicom_to_png(dicom_path, output_path):
    """Convert a single DICOM file to PNG format"""
    try:
        import pydicom
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array
        
        if img.max() > 0:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        Image.fromarray(img).save(output_path)
        return True
    except Exception as e:
        print(f"   Warning: Could not convert {dicom_path}: {e}")
        return False


def preprocess_rsna_dataset():
    """
    Convert RSNA DICOM files to PNG format
    
    Expected input structure:
        training_data/pneumonia/rsna/stage_2_train_images/*.dcm
        training_data/pneumonia/rsna/stage_2_train_labels.csv
    
    Output structure:
        training_data/pneumonia/rsna/images_png/*.png
        (Organized NORMAL/PNEUMONIA folders will be created by training script)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING RSNA DATASET (DICOM to PNG)")
    print("=" * 60)
    
    rsna_dir = os.path.join(PNEUMONIA_DATA_DIR, 'rsna')
    dicom_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    output_dir = os.path.join(rsna_dir, 'images_png')
    
    if not os.path.exists(dicom_dir):
        print(f"   [SKIP] RSNA DICOM folder not found: {dicom_dir}")
        print("   Download from: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data")
        return False
    
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 1000:
        print(f"   [SKIP] Already converted: {len(os.listdir(output_dir))} PNG files exist")
        return True
    
    os.makedirs(output_dir, exist_ok=True)
    
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    print(f"   Found {len(dicom_files)} DICOM files to convert")
    
    if len(dicom_files) == 0:
        print("   [SKIP] No DICOM files found")
        return False
    
    try:
        import pydicom
    except ImportError:
        print("   [ERROR] pydicom not installed. Install with: pip install pydicom")
        return False
    
    converted = 0
    failed = 0
    
    print("   Converting DICOM to PNG (this may take a while)...")
    
    def convert_single(dicom_file):
        dicom_path = os.path.join(dicom_dir, dicom_file)
        png_file = dicom_file.replace('.dcm', '.png')
        output_path = os.path.join(output_dir, png_file)
        
        if os.path.exists(output_path):
            return True
        
        return convert_dicom_to_png(dicom_path, output_path)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(convert_single, f): f for f in dicom_files}
        
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                converted += 1
            else:
                failed += 1
            
            if (i + 1) % 1000 == 0:
                print(f"   Progress: {i + 1}/{len(dicom_files)} files processed")
    
    print(f"   Conversion complete: {converted} converted, {failed} failed")
    return converted > 0


def preprocess_nih_dataset():
    """
    Extract pneumonia and normal cases from NIH ChestX-ray14 dataset
    
    Expected input structure:
        training_data/pneumonia/nih/images/*.png
        training_data/pneumonia/nih/Data_Entry_2017.csv
    
    Output structure:
        training_data/pneumonia/nih/organized/NORMAL/*.png
        training_data/pneumonia/nih/organized/PNEUMONIA/*.png
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING NIH CHESTX-RAY14 DATASET")
    print("=" * 60)
    
    nih_dir = os.path.join(PNEUMONIA_DATA_DIR, 'nih')
    images_dir = os.path.join(nih_dir, 'images')
    labels_file = os.path.join(nih_dir, 'Data_Entry_2017.csv')
    organized_dir = os.path.join(nih_dir, 'organized')
    
    if os.path.exists(organized_dir):
        normal_count = len(os.listdir(os.path.join(organized_dir, 'NORMAL'))) if os.path.exists(os.path.join(organized_dir, 'NORMAL')) else 0
        pneumonia_count = len(os.listdir(os.path.join(organized_dir, 'PNEUMONIA'))) if os.path.exists(os.path.join(organized_dir, 'PNEUMONIA')) else 0
        if normal_count > 100 and pneumonia_count > 100:
            print(f"   [SKIP] Already organized: {normal_count} normal, {pneumonia_count} pneumonia")
            return True
    
    if not os.path.exists(images_dir):
        print(f"   [SKIP] NIH images folder not found: {images_dir}")
        print("   Download from: https://www.kaggle.com/datasets/nih-chest-xrays/data")
        return False
    
    if not os.path.exists(labels_file):
        print(f"   [SKIP] NIH labels file not found: {labels_file}")
        return False
    
    print("   Loading NIH metadata...")
    df = pd.read_csv(labels_file)
    print(f"   Total records: {len(df)}")
    
    pneumonia_df = df[df['Finding Labels'].str.contains('Pneumonia', na=False)]
    normal_df = df[df['Finding Labels'] == 'No Finding']
    
    pneumonia_count = len(pneumonia_df)
    normal_count = len(normal_df)
    print(f"   Pneumonia cases: {pneumonia_count}")
    print(f"   Normal cases: {normal_count}")
    
    if normal_count > pneumonia_count * 3:
        sample_size = pneumonia_count * 3
        normal_df = normal_df.sample(n=sample_size, random_state=42)
        print(f"   Balancing classes: keeping all {pneumonia_count} pneumonia, sampling {sample_size} normal (3:1 max ratio)")
    else:
        print(f"   Classes balanced enough: keeping all samples")
    
    normal_out = os.path.join(organized_dir, 'NORMAL')
    pneumonia_out = os.path.join(organized_dir, 'PNEUMONIA')
    os.makedirs(normal_out, exist_ok=True)
    os.makedirs(pneumonia_out, exist_ok=True)
    
    print("   Copying pneumonia images...")
    pneumonia_copied = 0
    for _, row in pneumonia_df.iterrows():
        src = os.path.join(images_dir, row['Image Index'])
        dst = os.path.join(pneumonia_out, row['Image Index'])
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            pneumonia_copied += 1
        if pneumonia_copied % 500 == 0 and pneumonia_copied > 0:
            print(f"      Progress: {pneumonia_copied}/{len(pneumonia_df)}")
    
    print("   Copying normal images...")
    normal_copied = 0
    for _, row in normal_df.iterrows():
        src = os.path.join(images_dir, row['Image Index'])
        dst = os.path.join(normal_out, row['Image Index'])
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            normal_copied += 1
        if normal_copied % 500 == 0 and normal_copied > 0:
            print(f"      Progress: {normal_copied}/{len(normal_df)}")
    
    print(f"   Complete: {pneumonia_copied} pneumonia, {normal_copied} normal images organized")
    return pneumonia_copied > 0 or normal_copied > 0


def preprocess_nih_resized_dataset():
    """
    Extract pneumonia and normal cases from NIH Resized 224x224 dataset
    
    Expected input structure:
        training_data/pneumonia/nih_resized/images/*.png (or direct images)
        training_data/pneumonia/nih_resized/Data_Entry_2017.csv
    
    Output structure:
        training_data/pneumonia/nih_resized/organized/NORMAL/*.png
        training_data/pneumonia/nih_resized/organized/PNEUMONIA/*.png
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING NIH RESIZED 224x224 DATASET")
    print("=" * 60)
    
    nih_resized_paths = [
        os.path.join(PNEUMONIA_DATA_DIR, 'nih_resized'),
        os.path.join(PNEUMONIA_DATA_DIR, 'nih_224x224'),
        os.path.join(PNEUMONIA_DATA_DIR, 'nih-chest-x-ray-14-224x224-resized'),
    ]
    
    nih_resized_dir = None
    for path in nih_resized_paths:
        if os.path.exists(path):
            nih_resized_dir = path
            break
    
    if nih_resized_dir is None:
        print("   [SKIP] NIH Resized dataset not found")
        print("   Download from: https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized")
        return False
    
    organized_dir = os.path.join(nih_resized_dir, 'organized')
    if os.path.exists(organized_dir):
        normal_count = len(os.listdir(os.path.join(organized_dir, 'NORMAL'))) if os.path.exists(os.path.join(organized_dir, 'NORMAL')) else 0
        pneumonia_count = len(os.listdir(os.path.join(organized_dir, 'PNEUMONIA'))) if os.path.exists(os.path.join(organized_dir, 'PNEUMONIA')) else 0
        if normal_count > 100 and pneumonia_count > 100:
            print(f"   [SKIP] Already organized: {normal_count} normal, {pneumonia_count} pneumonia")
            return True
    
    images_dir = os.path.join(nih_resized_dir, 'images')
    if not os.path.exists(images_dir):
        images_dir = nih_resized_dir
    
    labels_file = None
    for csv_name in ['Data_Entry_2017.csv', 'labels.csv', 'data_entry.csv']:
        csv_path = os.path.join(nih_resized_dir, csv_name)
        if os.path.exists(csv_path):
            labels_file = csv_path
            break
    
    if labels_file is None:
        print("   [SKIP] Labels file not found")
        return False
    
    print(f"   Loading labels from: {labels_file}")
    df = pd.read_csv(labels_file)
    print(f"   Total records: {len(df)}")
    
    pneumonia_df = df[df['Finding Labels'].str.contains('Pneumonia', na=False)]
    normal_df = df[df['Finding Labels'] == 'No Finding']
    
    pneumonia_count = len(pneumonia_df)
    normal_count = len(normal_df)
    print(f"   Pneumonia cases: {pneumonia_count}")
    print(f"   Normal cases: {normal_count}")
    
    if normal_count > pneumonia_count * 3:
        sample_size = pneumonia_count * 3
        normal_df = normal_df.sample(n=sample_size, random_state=42)
        print(f"   Balancing classes: keeping all {pneumonia_count} pneumonia, sampling {sample_size} normal (3:1 max ratio)")
    else:
        print(f"   Classes balanced enough: keeping all samples")
    
    normal_out = os.path.join(organized_dir, 'NORMAL')
    pneumonia_out = os.path.join(organized_dir, 'PNEUMONIA')
    os.makedirs(normal_out, exist_ok=True)
    os.makedirs(pneumonia_out, exist_ok=True)
    
    print("   Copying pneumonia images...")
    pneumonia_copied = 0
    for _, row in pneumonia_df.iterrows():
        src = os.path.join(images_dir, row['Image Index'])
        dst = os.path.join(pneumonia_out, row['Image Index'])
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            pneumonia_copied += 1
    
    print("   Copying normal images...")
    normal_copied = 0
    for _, row in normal_df.iterrows():
        src = os.path.join(images_dir, row['Image Index'])
        dst = os.path.join(normal_out, row['Image Index'])
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            normal_copied += 1
    
    print(f"   Complete: {pneumonia_copied} pneumonia, {normal_copied} normal images organized")
    return pneumonia_copied > 0 or normal_copied > 0


def preprocess_vindr_dataset():
    """
    Convert VinDr-CXR DICOM files and organize by labels
    
    Expected input structure:
        training_data/pneumonia/vindr/train/*.dicom
        training_data/pneumonia/vindr/image_labels_train.csv
    
    Output structure:
        training_data/pneumonia/vindr/NORMAL/*.png
        training_data/pneumonia/vindr/PNEUMONIA/*.png
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING VINDR-CXR DATASET")
    print("=" * 60)
    
    vindr_dir = os.path.join(PNEUMONIA_DATA_DIR, 'vindr')
    
    if not os.path.exists(vindr_dir):
        print("   [SKIP] VinDr-CXR dataset not found")
        print("   Download from: https://physionet.org/content/vindr-cxr/1.0.0/")
        return False
    
    normal_out = os.path.join(vindr_dir, 'NORMAL')
    pneumonia_out = os.path.join(vindr_dir, 'PNEUMONIA')
    
    if os.path.exists(normal_out) and os.path.exists(pneumonia_out):
        normal_count = len(os.listdir(normal_out))
        pneumonia_count = len(os.listdir(pneumonia_out))
        if normal_count > 100 and pneumonia_count > 100:
            print(f"   [SKIP] Already organized: {normal_count} normal, {pneumonia_count} pneumonia")
            return True
    
    labels_file = os.path.join(vindr_dir, 'image_labels_train.csv')
    if not os.path.exists(labels_file):
        labels_file = os.path.join(vindr_dir, 'annotations_train.csv')
    
    if not os.path.exists(labels_file):
        print("   [SKIP] VinDr labels file not found")
        return False
    
    train_dir = os.path.join(vindr_dir, 'train')
    if not os.path.exists(train_dir):
        print("   [SKIP] VinDr train folder not found")
        return False
    
    try:
        import pydicom
    except ImportError:
        print("   [ERROR] pydicom not installed. Install with: pip install pydicom")
        return False
    
    print("   Loading VinDr labels...")
    df = pd.read_csv(labels_file)
    
    pneumonia_related = ['Pneumonia', 'Consolidation', 'Infiltration', 'Lung Opacity']
    
    os.makedirs(normal_out, exist_ok=True)
    os.makedirs(pneumonia_out, exist_ok=True)
    
    print("   Processing and converting images...")
    pneumonia_count = 0
    normal_count = 0
    
    for _, row in df.iterrows():
        image_id = row.get('image_id', row.get('Image Index', ''))
        labels = str(row.get('labels', row.get('Finding Labels', '')))
        
        is_pneumonia = any(label in labels for label in pneumonia_related)
        is_normal = 'No finding' in labels or 'No Finding' in labels
        
        if not is_pneumonia and not is_normal:
            continue
        
        dicom_path = os.path.join(train_dir, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            dicom_path = os.path.join(train_dir, image_id)
        
        if not os.path.exists(dicom_path):
            continue
        
        output_class = 'PNEUMONIA' if is_pneumonia else 'NORMAL'
        output_path = os.path.join(vindr_dir, output_class, f"{image_id}.png")
        
        if not os.path.exists(output_path):
            if convert_dicom_to_png(dicom_path, output_path):
                if is_pneumonia:
                    pneumonia_count += 1
                else:
                    normal_count += 1
    
    print(f"   Complete: {pneumonia_count} pneumonia, {normal_count} normal images processed")
    return pneumonia_count > 0 or normal_count > 0


def preprocess_chexpert_dataset():
    """
    Organize CheXpert dataset by extracting pneumonia-related and normal cases
    
    Expected input structure:
        training_data/pneumonia/chexpert/train/patient*/study*/*.jpg
        training_data/pneumonia/chexpert/train.csv
    
    Output structure:
        training_data/pneumonia/chexpert/organized/NORMAL/*.jpg
        training_data/pneumonia/chexpert/organized/PNEUMONIA/*.jpg
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING CHEXPERT DATASET")
    print("=" * 60)
    
    chexpert_dir = os.path.join(PNEUMONIA_DATA_DIR, 'chexpert')
    
    if not os.path.exists(chexpert_dir):
        print("   [SKIP] CheXpert dataset not found")
        print("   Register at: https://stanfordmlgroup.github.io/competitions/chexpert/")
        return False
    
    organized_dir = os.path.join(chexpert_dir, 'organized')
    if os.path.exists(organized_dir):
        normal_count = len(os.listdir(os.path.join(organized_dir, 'NORMAL'))) if os.path.exists(os.path.join(organized_dir, 'NORMAL')) else 0
        pneumonia_count = len(os.listdir(os.path.join(organized_dir, 'PNEUMONIA'))) if os.path.exists(os.path.join(organized_dir, 'PNEUMONIA')) else 0
        if normal_count > 100 and pneumonia_count > 100:
            print(f"   [SKIP] Already organized: {normal_count} normal, {pneumonia_count} pneumonia")
            return True
    
    labels_file = os.path.join(chexpert_dir, 'train.csv')
    if not os.path.exists(labels_file):
        print("   [SKIP] CheXpert labels file (train.csv) not found")
        return False
    
    print("   Loading CheXpert labels...")
    df = pd.read_csv(labels_file)
    print(f"   Total records: {len(df)}")
    
    pneumonia_cols = ['Pneumonia', 'Consolidation', 'Lung Opacity']
    available_cols = [col for col in pneumonia_cols if col in df.columns]
    
    if not available_cols:
        print("   [SKIP] No pneumonia-related columns found in CheXpert CSV")
        return False
    
    print(f"   Using columns: {available_cols}")
    
    pneumonia_mask = df[available_cols].apply(lambda x: x == 1.0).any(axis=1)
    pneumonia_df = df[pneumonia_mask]
    
    condition_cols = [col for col in df.columns if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
    condition_cols = [col for col in condition_cols if col in df.columns]
    
    if condition_cols:
        normal_mask = df[condition_cols].apply(lambda x: (x == 0.0) | (x == -1.0) | (x.isna())).all(axis=1)
        normal_df = df[normal_mask]
    else:
        normal_df = pd.DataFrame()
    
    print(f"   Pneumonia cases: {len(pneumonia_df)}")
    print(f"   Normal cases: {len(normal_df)}")
    
    if len(pneumonia_df) == 0:
        print("   [SKIP] No pneumonia cases found")
        return False
    
    sample_size = min(len(pneumonia_df), len(normal_df)) if len(normal_df) > 0 else len(pneumonia_df)
    print(f"   Using {sample_size} samples per class")
    
    if len(normal_df) > sample_size:
        normal_df = normal_df.sample(n=sample_size, random_state=42)
    if len(pneumonia_df) > sample_size:
        pneumonia_df = pneumonia_df.sample(n=sample_size, random_state=42)
    
    normal_out = os.path.join(organized_dir, 'NORMAL')
    pneumonia_out = os.path.join(organized_dir, 'PNEUMONIA')
    os.makedirs(normal_out, exist_ok=True)
    os.makedirs(pneumonia_out, exist_ok=True)
    
    base_dir = os.path.dirname(labels_file)
    
    print("   Copying pneumonia images...")
    pneumonia_copied = 0
    for _, row in pneumonia_df.iterrows():
        img_path = os.path.join(base_dir, row['Path'])
        if os.path.exists(img_path):
            filename = os.path.basename(img_path)
            patient_id = row['Path'].split('/')[1] if '/' in row['Path'] else 'unknown'
            dst = os.path.join(pneumonia_out, f"{patient_id}_{filename}")
            if not os.path.exists(dst):
                shutil.copy(img_path, dst)
                pneumonia_copied += 1
    
    print("   Copying normal images...")
    normal_copied = 0
    for _, row in normal_df.iterrows():
        img_path = os.path.join(base_dir, row['Path'])
        if os.path.exists(img_path):
            filename = os.path.basename(img_path)
            patient_id = row['Path'].split('/')[1] if '/' in row['Path'] else 'unknown'
            dst = os.path.join(normal_out, f"{patient_id}_{filename}")
            if not os.path.exists(dst):
                shutil.copy(img_path, dst)
                normal_copied += 1
    
    print(f"   Complete: {pneumonia_copied} pneumonia, {normal_copied} normal images organized")
    return pneumonia_copied > 0 or normal_copied > 0


def check_all_datasets():
    """Check status of all pneumonia datasets"""
    print("\n" + "=" * 60)
    print("PNEUMONIA DATASET STATUS CHECK")
    print("=" * 60)
    
    datasets = [
        {
            'name': 'Kaggle Chest X-Ray',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'kaggle', 'train'),
            'alt_path': os.path.join(PNEUMONIA_DATA_DIR, 'train'),
            'type': 'organized',
            'download': 'https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia'
        },
        {
            'name': 'RSNA Pneumonia',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'rsna', 'images_png'),
            'raw_path': os.path.join(PNEUMONIA_DATA_DIR, 'rsna', 'stage_2_train_images'),
            'type': 'rsna',
            'download': 'https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data'
        },
        {
            'name': 'NIH ChestX-ray14',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'nih', 'organized'),
            'raw_path': os.path.join(PNEUMONIA_DATA_DIR, 'nih', 'images'),
            'type': 'nih',
            'download': 'https://www.kaggle.com/datasets/nih-chest-xrays/data'
        },
        {
            'name': 'COVID-Pneumonia-Normal',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'covid_pneumonia_normal'),
            'type': 'organized',
            'download': 'https://data.mendeley.com/datasets/dvntn9yhd2/1'
        },
        {
            'name': 'Roboflow',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'roboflow', 'train'),
            'type': 'organized',
            'download': 'https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia'
        },
        {
            'name': 'NIH Resized 224x224',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'nih_resized', 'organized'),
            'raw_path': os.path.join(PNEUMONIA_DATA_DIR, 'nih_resized'),
            'type': 'nih',
            'download': 'https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized'
        },
        {
            'name': 'VinDr-CXR',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'vindr'),
            'type': 'vindr',
            'download': 'https://physionet.org/content/vindr-cxr/1.0.0/'
        },
        {
            'name': 'CheXpert',
            'path': os.path.join(PNEUMONIA_DATA_DIR, 'chexpert', 'organized'),
            'raw_path': os.path.join(PNEUMONIA_DATA_DIR, 'chexpert', 'train'),
            'type': 'chexpert',
            'download': 'https://stanfordmlgroup.github.io/competitions/chexpert/'
        },
    ]
    
    ready_datasets = []
    needs_preprocessing = []
    not_downloaded = []
    
    for dataset in datasets:
        path = dataset['path']
        alt_path = dataset.get('alt_path')
        raw_path = dataset.get('raw_path')
        
        if os.path.exists(path):
            normal_path = os.path.join(path, 'NORMAL')
            pneumonia_path = os.path.join(path, 'PNEUMONIA')
            
            if os.path.exists(normal_path) and os.path.exists(pneumonia_path):
                normal_count = len([f for f in os.listdir(normal_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                pneumonia_count = len([f for f in os.listdir(pneumonia_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                if normal_count > 0 and pneumonia_count > 0:
                    dataset['status'] = 'ready'
                    dataset['normal'] = normal_count
                    dataset['pneumonia'] = pneumonia_count
                    ready_datasets.append(dataset)
                    continue
        
        if alt_path and os.path.exists(alt_path):
            normal_path = os.path.join(alt_path, 'NORMAL')
            pneumonia_path = os.path.join(alt_path, 'PNEUMONIA')
            
            if os.path.exists(normal_path) and os.path.exists(pneumonia_path):
                normal_count = len([f for f in os.listdir(normal_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                pneumonia_count = len([f for f in os.listdir(pneumonia_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                
                if normal_count > 0 and pneumonia_count > 0:
                    dataset['status'] = 'ready'
                    dataset['normal'] = normal_count
                    dataset['pneumonia'] = pneumonia_count
                    ready_datasets.append(dataset)
                    continue
        
        if raw_path and os.path.exists(raw_path):
            dataset['status'] = 'needs_preprocessing'
            needs_preprocessing.append(dataset)
        else:
            dataset['status'] = 'not_downloaded'
            not_downloaded.append(dataset)
    
    print("\n[READY FOR TRAINING]")
    if ready_datasets:
        for ds in ready_datasets:
            print(f"   [OK] {ds['name']}: {ds['normal']} normal, {ds['pneumonia']} pneumonia")
    else:
        print("   No datasets ready")
    
    print("\n[NEEDS PREPROCESSING]")
    if needs_preprocessing:
        for ds in needs_preprocessing:
            print(f"   [!] {ds['name']}: Raw data found, run preprocessing")
    else:
        print("   None")
    
    print("\n[NOT DOWNLOADED]")
    if not_downloaded:
        for ds in not_downloaded:
            print(f"   [-] {ds['name']}")
            print(f"       Download: {ds['download']}")
    else:
        print("   None")
    
    return ready_datasets, needs_preprocessing, not_downloaded


def run_all_preprocessing():
    """Run all preprocessing tasks"""
    print("\n" + "=" * 60)
    print("RUNNING ALL PREPROCESSING TASKS")
    print("=" * 60)
    
    results = {
        'rsna': preprocess_rsna_dataset(),
        'nih': preprocess_nih_dataset(),
        'nih_resized': preprocess_nih_resized_dataset(),
        'vindr': preprocess_vindr_dataset(),
        'chexpert': preprocess_chexpert_dataset(),
    }
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "[OK]" if success else "[SKIP]"
        print(f"   {status} {name}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PNEUMONIA DATASET PREPROCESSING TOOL")
    print("=" * 60)
    
    check_all_datasets()
    
    run_all_preprocessing()
    
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    check_all_datasets()
    
    print("\n" + "=" * 60)
    print("NEXT STEP: Run training")
    print("=" * 60)
    print("   python training_scripts/train_pneumonia_models.py")
    print("=" * 60)

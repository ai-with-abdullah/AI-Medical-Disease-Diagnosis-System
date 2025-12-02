# Skin Cancer Dataset Setup (Multi-Dataset Support)

This folder supports **6 different skin cancer datasets**. You can use any combination - the training script auto-detects and combines all available data for better accuracy!

## Supported Datasets

| # | Dataset | Images | Size | Accuracy | Best For |
|---|---------|--------|------|----------|----------|
| 1 | HAM10000 | 10,015 | 2.7 GB | 88-93% | Quick start |
| 2 | ISIC 2019 | 25,331 | 9 GB | 90-95% | High accuracy |
| 3 | ISIC 2020 | 33,126 | 15 GB | 85-92% | Melanoma focus |
| 4 | PAD-UFES-20 | 2,298 | 500 MB | 85-90% | Smartphone images |
| 5 | Melanoma Binary | 10,605 | 3 GB | 88-93% | Binary classification |
| 6 | Pre-organized | Any | Any | Varies | Custom datasets |

## Recommended Strategies

### Strategy A: Quick Start (88-92% Accuracy)
Download HAM10000 only.
**Total: ~10,000 images, ~2.7 GB**

### Strategy B: Best Accuracy (92-95% Accuracy)
Download HAM10000 + ISIC 2019 + PAD-UFES-20.
**Total: ~37,600 images, ~12 GB**

### Strategy C: Maximum Accuracy (94-97% Accuracy)
Download all datasets.
**Total: 80,000+ images, ~30 GB**

## Dataset Download Links

### 1. HAM10000 (Recommended Start)
- **Link:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Size:** 2.7 GB (10,015 dermoscopy images)
- **Classes:** 7 types of skin lesions
- **Place in:** `training_data/skin_cancer/ham10000/`

### 2. ISIC 2019 Challenge (High Accuracy)
- **Link:** https://www.kaggle.com/datasets/andrewmvd/isic-2019
- **Size:** 9 GB (25,331 images)
- **Classes:** 8 diagnostic categories
- **Place in:** `training_data/skin_cancer/isic2019/`

### 3. ISIC 2020 Challenge (Melanoma Focus)
- **Link:** https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data
- **Size:** 15 GB (33,126 images)
- **Classes:** Binary (Benign/Malignant)
- **Place in:** `training_data/skin_cancer/isic2020/`

### 4. PAD-UFES-20 (Smartphone Images)
- **Link:** https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
- **Size:** 500 MB (2,298 images)
- **Classes:** 6 lesion types
- **Place in:** `training_data/skin_cancer/pad_ufes_20/`

### 5. Melanoma Binary Dataset
- **Link:** https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- **Size:** 3 GB (10,605 images)
- **Classes:** 2 (Benign, Malignant)
- **Place in:** `training_data/skin_cancer/melanoma_binary/`

## Folder Structure

```
training_data/skin_cancer/
|
+-- ham10000/                          <- Dataset 1: HAM10000
|   +-- HAM10000_images_part_1/
|   |   +-- ISIC_0024306.jpg
|   |   +-- ... (5,000+ images)
|   +-- HAM10000_images_part_2/
|   |   +-- ... (5,000+ images)
|   +-- HAM10000_metadata.csv          <- Labels (REQUIRED)
|
+-- isic2019/                          <- Dataset 2: ISIC 2019
|   +-- ISIC_2019_Training_Input/
|   |   +-- ISIC_0000000.jpg
|   |   +-- ... (25,331 images)
|   +-- ISIC_2019_Training_GroundTruth.csv
|
+-- isic2020/                          <- Dataset 3: ISIC 2020
|   +-- train/
|   |   +-- *.jpg (33,126 images)
|   +-- train.csv
|
+-- pad_ufes_20/                       <- Dataset 4: PAD-UFES-20
|   +-- images/
|   |   +-- *.png (2,298 images)
|   +-- metadata.csv
|
+-- melanoma_binary/                   <- Dataset 5: Melanoma Binary
|   +-- benign/
|   |   +-- ... (benign images)
|   +-- malignant/
|       +-- ... (malignant images)
|
+-- organized/                         <- Auto-generated: Combined dataset
    +-- nv/
    +-- mel/
    +-- bkl/
    +-- bcc/
    +-- akiec/
    +-- vasc/
    +-- df/
```

## Training

After placing datasets in the correct folders, run:

```bash
python training_scripts/train_skin_model.py
```

The script automatically:
1. Detects all available datasets
2. Combines data into unified class folders
3. Applies class balancing
4. Trains ResNet50 with transfer learning
5. Saves model to `models/weights/skin_resnet50.h5`

## Skin Cancer Classes (7 Types)

| Code | Disease Name | Category | Risk Level |
|------|-------------|----------|------------|
| nv | Melanocytic Nevus (Mole) | Benign | Low |
| mel | Melanoma | Malignant | HIGH - Urgent |
| bkl | Benign Keratosis | Benign | Low |
| bcc | Basal Cell Carcinoma | Malignant | Medium-High |
| akiec | Actinic Keratosis | Pre-cancerous | Medium |
| vasc | Vascular Lesion | Vascular | Low |
| df | Dermatofibroma | Benign | Low |

## Expected Results

| Data Size | Training Time | Accuracy |
|-----------|--------------|----------|
| ~10,000 images | 20-40 min | 88-92% |
| ~25,000 images | 40-60 min | 90-94% |
| ~40,000 images | 1-2 hours | 93-96% |
| ~80,000+ images | 2-4 hours | 95-97% |

## Troubleshooting

### "No dataset found" Error
- Ensure at least one dataset folder exists
- Check that metadata CSV files are present
- Verify image folders contain .jpg/.png files

### "TensorFlow not installed" Error
```bash
pip install tensorflow
```

### Training is Very Slow
- Use GPU if available (CUDA)
- Reduce batch size in script
- Start with smaller dataset (HAM10000 only)

## Team Members

- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz

# Skin Disease Dataset Setup

## Supported Datasets

### 1. HAM10000 (Recommended)
- **Link:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Size:** 2.7 GB, 10,015 images
- **Classes:** 7 types of skin lesions
- **Best for:** General skin disease detection

### 2. ISIC Archive (Alternative)
- **Link:** https://www.isic-archive.com/
- **Size:** Various
- **Note:** Pre-organized in class folders

### 3. DermNet (Alternative)  
- **Link:** https://www.kaggle.com/datasets/shubhamgoel27/dermnet
- **Size:** 19,500 images, 23 classes
- **Note:** Pre-organized in class folders

## Quick Setup (HAM10000)

### Step 1: Download from Kaggle
1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download" (requires free Kaggle account)
3. Download: archive.zip (about 2.7 GB)

### Step 2: Extract Files
After extracting archive.zip, you should have:
```
archive/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── HAM10000_metadata.csv
└── hmnist_28_28_RGB.csv (optional)
```

### Step 3: Copy to This Folder
Copy these files/folders here:
```
training_data/skin_disease/
├── HAM10000_images_part_1/   <- Copy this folder
├── HAM10000_images_part_2/   <- Copy this folder
└── HAM10000_metadata.csv     <- Copy this file (REQUIRED)
```

### Step 4: Run Training
```bash
python training_scripts/train_skin_model.py
```

The script will automatically:
- Detect the HAM10000 dataset
- Organize images into class folders
- Train the ResNet50 model
- Save to models/weights/skin_resnet50.h5

## HAM10000 Classes

| Code | Full Name | Description |
|------|-----------|-------------|
| nv | Melanocytic Nevus | Common mole (benign) |
| mel | Melanoma | Skin cancer (malignant) |
| bkl | Benign Keratosis | Age spots, seborrheic keratosis |
| bcc | Basal Cell Carcinoma | Common skin cancer |
| akiec | Actinic Keratosis | Pre-cancerous lesion |
| vasc | Vascular Lesion | Blood vessel abnormalities |
| df | Dermatofibroma | Benign fibrous nodule |

## Expected Training Results

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 88-93% |
| Training Time | 20-40 minutes |
| Model Size | ~100 MB |

## Alternative: Pre-organized Datasets

If using ISIC or DermNet, organize images in class folders:
```
training_data/skin_disease/
└── organized/
    ├── melanoma/
    │   ├── image1.jpg
    │   └── ...
    ├── nevus/
    │   └── ...
    └── other_class/
        └── ...
```

The training script will auto-detect this structure.

## Troubleshooting

### "No dataset found" Error
- Ensure HAM10000_metadata.csv is in this folder
- Ensure image folders (HAM10000_images_part_1, etc.) exist
- Check file permissions

### "TensorFlow not installed" Error
```bash
pip install tensorflow
```

### Training is Very Slow
- Consider using GPU (CUDA)
- Reduce batch size in script if memory issues
- Use smaller dataset subset for testing

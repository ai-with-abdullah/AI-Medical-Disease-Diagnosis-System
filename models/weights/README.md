# Trained Model Weights

This folder stores trained model weights after running the training scripts.

## Expected Files After Training

### Heart Disease Models (Random Forest) - 3 Disease Types

The system supports **3 specialized heart disease prediction models**:

| # | Disease Type | Icon | Description |
|---|--------------|------|-------------|
| 1 | Generic Cardiovascular Disease | ❤️ | General heart disease risk assessment (Yes/No) |
| 2 | Coronary Artery Disease (CAD) | 💔 | Blockage in heart arteries - specific CAD detection |
| 3 | Cardiac Arrhythmia | 📈 | Irregular heartbeat detection and classification |

#### Model Files:
| File | Description | Training Script |
|------|-------------|-----------------|
| `heart_generic_model.pkl` | Generic CVD prediction model | `train_heart_models.py` |
| `heart_generic_scaler.pkl` | Feature scaler for generic model | `train_heart_models.py` |
| `heart_cad_model.pkl` | Coronary Artery Disease model | `train_heart_models.py` |
| `heart_cad_scaler.pkl` | Feature scaler for CAD model | `train_heart_models.py` |
| `heart_arrhythmia_model.pkl` | Cardiac Arrhythmia model | `train_heart_models.py` |
| `heart_arrhythmia_scaler.pkl` | Feature scaler for arrhythmia model | `train_heart_models.py` |

#### How Training Works:
- **Generic CVD & CAD Models:** Both trained on the same combined heart disease data (all CSVs from training_data/heart_disease/ merged together)
- **Arrhythmia Model:** Trained separately on UCI Arrhythmia dataset only (training_data/arrhythmia/arrhythmia.data)

#### Key Dataset Sources:
- **Heart Disease (Combined):** Multiple Kaggle datasets including https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Arrhythmia:** https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data

### Pneumonia Detection Models (CNN)
| File | Description | Training Script |
|------|-------------|-----------------|
| `pneumonia_resnet50.h5` | ResNet50-based pneumonia detector | `train_pneumonia_models.py` |
| `pneumonia_efficientnet.h5` | EfficientNet-based pneumonia detector | `train_pneumonia_models.py` |
| `pneumonia_mobilenet.h5` | MobileNet-based pneumonia detector | `train_pneumonia_models.py` |

### Skin Cancer Detection Model (CNN) - Multi-Dataset Support
| File | Description | Training Script |
|------|-------------|-----------------|
| `skin_resnet50.h5` | ResNet50-based skin cancer classifier (7 classes) | `train_skin_model.py` |
| `skin_classes.json` | Class mapping information | `train_skin_model.py` |

**Supported Datasets:** HAM10000, ISIC 2019, ISIC 2020, PAD-UFES-20, Melanoma Binary, pre-organized

## Training Instructions

1. **Download datasets** - Follow instructions in `COMPREHENSIVE_TRAINING_GUIDE.md`
2. **Prepare data** - Run `python training_scripts/prepare_training_data.py`
3. **Train models** - Run individual training scripts:
   - `python training_scripts/train_heart_models.py`
   - `python training_scripts/train_pneumonia_models.py`
   - `python training_scripts/train_skin_model.py`

## Color Blindness Module

**Note:** The Color Blindness module does NOT require training. It uses interactive clinical tests with predefined correct answers based on medical standards. Test images are stored in the `assets/` folder.

## Auto-Detection

The application automatically detects trained models:
- **DEMO MODE**: When weights are not present (uses simulated predictions)
- **PRODUCTION MODE**: When weights are present (uses real AI predictions)

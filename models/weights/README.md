# Trained Model Weights

This folder stores trained model weights after running the training scripts.

## Expected Files After Training

### Heart Disease Models (Random Forest)
| File | Description | Training Script |
|------|-------------|-----------------|
| `heart_generic_model.pkl` | Generic CVD prediction model | `train_heart_models.py` |
| `heart_generic_scaler.pkl` | Feature scaler for generic model | `train_heart_models.py` |
| `heart_cad_model.pkl` | Coronary Artery Disease model | `train_heart_models.py` |
| `heart_cad_scaler.pkl` | Feature scaler for CAD model | `train_heart_models.py` |
| `heart_arrhythmia_model.pkl` | Cardiac Arrhythmia model | `train_heart_models.py` |
| `heart_arrhythmia_scaler.pkl` | Feature scaler for arrhythmia model | `train_heart_models.py` |

### Pneumonia Detection Models (CNN)
| File | Description | Training Script |
|------|-------------|-----------------|
| `pneumonia_resnet50.h5` | ResNet50-based pneumonia detector | `train_pneumonia_models.py` |
| `pneumonia_efficientnet.h5` | EfficientNet-based pneumonia detector | `train_pneumonia_models.py` |
| `pneumonia_mobilenet.h5` | MobileNet-based pneumonia detector | `train_pneumonia_models.py` |

### Skin Disease Detection Model (CNN)
| File | Description | Training Script |
|------|-------------|-----------------|
| `skin_resnet50.h5` | ResNet50-based skin disease classifier (7 classes) | `train_skin_model.py` |

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

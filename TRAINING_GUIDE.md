# Training Guide - Converting Demo to Production Models

## ⚠️ IMPORTANT: Current Implementation Status

The current application runs in **DEMO MODE** with simulated predictions. To make this expo-ready with real AI models, you must:

1. Collect real medical datasets (as outlined below)
2. Train the models using the provided architectures
3. Save trained model weights
4. Update model loading to use trained weights instead of demo mode

## 📊 Datasets Used for Training

### 1. Pneumonia Detection (X-Ray Images)
**Primary Dataset:**
- **Kaggle: "Chest X-Ray Images (Pneumonia)"**
  - Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - Size: 5,856 JPEG images
  - Classes: NORMAL (1,583 images) vs PNEUMONIA (4,273 images)
  - Resolution: Various sizes, resized to 224x224 for training
  - Pre-split: train/val/test folders provided

**Additional Datasets:**
- **NIH Chest X-Ray Dataset** - 112,120 frontal-view X-ray images
- **RSNA Pneumonia Detection Challenge** - 30,000 chest X-rays with bounding boxes

**Training Strategy:**
- 5-fold cross-validation using multiple model architectures
- Data augmentation: rotation (±15°), zoom (0.9-1.1), horizontal flip
- Models: ResNet50, EfficientNetB0, MobileNetV2

### 2. Pneumonia Audio (Cough/Breathing Sounds)
**Primary Dataset:**
- **Coswara COVID-19 Dataset**
  - Link: https://github.com/iiscleap/Coswara-Data
  - Size: 2,000+ audio samples
  - Types: Cough, breathing, voice
  - Format: WAV files, 44.1 kHz sampling rate

**Additional Sources:**
- **ESC-50 Environmental Sound Classification** - Background and cough sounds
- **FluSense Dataset** - Respiratory sound analysis

**Feature Extraction:**
- MFCC (Mel-Frequency Cepstral Coefficients) - 40 coefficients
- Spectrograms converted to images for CNN processing
- Audio length: 5 seconds (padded/truncated)

### 3. Skin Disease Images
**Primary Dataset:**
- **HAM10000 (Human Against Machine with 10000 training images)**
  - Link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
  - Size: 10,015 dermatoscopic images
  - Classes: 7 types of skin lesions
  - Resolution: 600x450 pixels (resized to 224x224)

**Classes:**
1. Melanocytic nevi (nv) - 6,705 images
2. Melanoma (mel) - 1,113 images  
3. Benign keratosis (bkl) - 1,099 images
4. Basal cell carcinoma (bcc) - 514 images
5. Actinic keratoses (akiec) - 327 images
6. Vascular lesions (vasc) - 142 images
7. Dermatofibroma (df) - 115 images

**Additional Dataset:**
- **ISIC 2019 Challenge** - 25,331 images across 8 diagnostic categories
- **DermNet Database** - Additional skin condition images

**Training Strategy:**
- Handle class imbalance with weighted loss function
- Data augmentation: rotation, flip, brightness adjustment, contrast
- Transfer learning from ImageNet pre-trained weights

### 4. Heart Disease Clinical Data
**Primary Dataset:**
- **UCI Heart Disease Dataset (Cleveland)**
  - Link: https://archive.ics.uci.edu/ml/datasets/heart+disease
  - Size: 303 patient records
  - Features: 14 clinical attributes
  - Target: 0 (no disease) vs 1-4 (disease severity levels)

**Features:**
1. Age (years)
2. Sex (1=male, 0=female)
3. Chest pain type (4 values)
4. Resting blood pressure (mm Hg)
5. Serum cholesterol (mg/dl)
6. Fasting blood sugar > 120 mg/dl (1=true, 0=false)
7. Resting ECG results (0,1,2)
8. Maximum heart rate achieved
9. Exercise induced angina (1=yes, 0=no)
10. ST depression induced by exercise
11. Slope of peak exercise ST segment
12. Number of major vessels (0-3) colored by fluoroscopy
13. Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)
14. Target: Diagnosis (0=no disease, 1=disease)

**Additional Datasets:**
- **Framingham Heart Study** - Expanded dataset with 4,240 records
- **Statlog Heart Disease Dataset** - 270 instances

**Training Strategy:**
- Random Forest Classifier with 100 estimators
- Feature scaling with StandardScaler
- 80/20 train-test split
- Cross-validation: 5-fold

### 5. Color Blindness & Eye Tests
**Primary Sources:**
- **Ishihara Test Plates**: Digitized standard 38-plate set
- **Synthetic Color Vision Tests**: Generated using Python/PIL

**Test Types & Datasets:**
1. **Ishihara Plates** - 38 standard plates digitized
2. **Farnsworth D-15** - 15 color caps arrangement
3. **Cambridge Color Test** - Synthetic chromatic contrast patterns
4. **Color Spectrum** - Generated gradient discrimination tests
5. **Anomaloscope Simulation** - RGB mixing simulation (Nagel-type)
6. **Snellen Chart** - Standard visual acuity chart (E, F, P, T, O, Z, L, C, D)
7. **Eye Muscle Tests** - Interactive convergence and tracking tests

**Generation Method:**
- Python libraries: PIL, NumPy, OpenCV
- Color spaces: RGB, HSV for precise color manipulation
- Validation against clinical color vision standards

## 🔧 Training Steps

### Step 1: Download and Organize Your Data

**1.1 Download Datasets:**
```bash
# Pneumonia X-Ray (Kaggle)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/pneumonia_xray/

# HAM10000 Skin Cancer (Kaggle)  
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/skin_disease/

# UCI Heart Disease
wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data -O data/heart_disease/heart_data.csv

# Coswara Audio (GitHub)
git clone https://github.com/iiscleap/Coswara-Data.git data/pneumonia_audio/
```

**1.2 Organize Directory Structure:**
```bash
data/
├── pneumonia_xray/
│   ├── train/
│   │   ├── NORMAL/          # 1,341 images
│   │   └── PNEUMONIA/       # 3,875 images
│   ├── val/
│   │   ├── NORMAL/          # 8 images
│   │   └── PNEUMONIA/       # 8 images
│   └── test/
│       ├── NORMAL/          # 234 images
│       └── PNEUMONIA/       # 390 images
├── pneumonia_audio/
│   ├── coswara/
│   │   ├── cough/           # WAV files
│   │   ├── breathing/
│   │   └── speech/
│   └── processed/
│       ├── spectrograms/    # Generated from audio
│       └── mfcc/            # MFCC features
├── skin_disease/
│   ├── images/              # 10,015 images
│   │   ├── nv/              # Melanocytic nevi (6,705)
│   │   ├── mel/             # Melanoma (1,113)
│   │   ├── bkl/             # Benign keratosis (1,099)
│   │   ├── bcc/             # Basal cell carcinoma (514)
│   │   ├── akiec/           # Actinic keratoses (327)
│   │   ├── vasc/            # Vascular lesions (142)
│   │   └── df/              # Dermatofibroma (115)
│   └── metadata.csv         # Labels and patient info
├── heart_disease/
│   ├── heart_data.csv       # 303 records (Cleveland)
│   └── framingham.csv       # 4,240 records (optional)
└── eye_tests/
    ├── ishihara_plates/     # 38 standard plates
    ├── farnsworth/          # 15 color caps
    ├── snellen_charts/      # Visual acuity charts
    └── synthetic/           # Generated test patterns
```

### Step 2: Train Pneumonia X-Ray Model
```python
# Update training/train_pneumonia.py
from training.train_models import ModelTrainer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your 5 datasets
datasets = []
for i in range(1, 6):
    datagen = ImageDataGenerator(rescale=1./255)
    ds = datagen.flow_from_directory(
        f'data/pneumonia_xray/dataset{i}',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    datasets.append(ds)

# Train using 5-dataset strategy
trainer = ModelTrainer(model_type='pneumonia')
model, history = trainer.five_dataset_training_strategy(datasets)

# Save trained model
model.save('models/weights/pneumonia_resnet50.h5')
```

### Step 3: Update Model Loading
**Current (Demo Mode):**
```python
# models/pneumonia_model.py - Line 45
def get_single_model_prediction(img_preprocessed, model_name):
    # Currently uses random predictions
    feature_score = np.random.random()
    ...
```

**Production Version:**
```python
# Load trained model once
TRAINED_MODELS = {
    'ResNet50': keras.models.load_model('models/weights/pneumonia_resnet50.h5'),
    'EfficientNet': keras.models.load_model('models/weights/pneumonia_efficientnet.h5'),
    'MobileNet': keras.models.load_model('models/weights/pneumonia_mobilenet.h5')
}

def get_single_model_prediction(img_preprocessed, model_name):
    model = TRAINED_MODELS[model_name]
    prediction_probs = model.predict(img_preprocessed)
    
    if prediction_probs[0][1] > 0.5:  # Index 1 = Pneumonia
        prediction = 'Pneumonia'
        confidence = prediction_probs[0][1]
    else:
        prediction = 'Normal'
        confidence = prediction_probs[0][0]
    
    return prediction, confidence
```

### Step 4: Train All Models

**For Skin Disease:**
```python
from training.train_models import ModelTrainer

trainer = ModelTrainer(model_type='skin')
# Load your 5 skin datasets
model, history = trainer.five_dataset_training_strategy(skin_datasets)
model.save('models/weights/skin_resnet50.h5')
```

**For Heart Disease:**
```python
from models.heart_model import train_heart_model
import pandas as pd

# Load heart disease dataset
data = pd.read_csv('data/heart_disease/heart_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Train Random Forest
model, scaler = train_heart_model(X, y)

# Save model and scaler
import joblib
joblib.dump(model, 'models/weights/heart_rf_model.pkl')
joblib.dump(scaler, 'models/weights/heart_scaler.pkl')
```

**For Audio Analysis:**
```python
# Train audio CNN
# Load audio features and labels
# Train model
# Save weights
audio_model.save('models/weights/audio_pneumonia_cnn.h5')
```

**For Color Blindness:**
```python
# Train 5 separate models (one per test type)
for test_type in ['ishihara', 'farnsworth', 'cambridge', 'spectrum', 'anomaloscope']:
    model = build_colorblind_cnn()
    # Train on respective dataset
    model.save(f'models/weights/colorblind_{test_type}.h5')
```

## 📝 Timeline for Your 1-Month Project

**Week 1: Data Collection & Preprocessing**
- Days 1-3: Collect datasets from Kaggle, UCI, etc.
- Days 4-5: Organize into 5-dataset structure
- Days 6-7: Preprocess and augment data

**Week 2: Model Training**
- Days 8-10: Train Pneumonia (X-ray) models
- Days 11-12: Train Skin disease models
- Days 13-14: Train Heart disease & Audio models

**Week 3: Color Blindness & Integration**
- Days 15-17: Train 5 color blindness models
- Days 18-19: Update model loading code
- Days 20-21: Test all integrations

**Week 4: Testing & Presentation Prep**
- Days 22-24: End-to-end testing
- Days 25-26: Create presentation slides
- Days 27-28: Practice demo
- Days 29-30: Buffer for fixes and final prep

## 🎯 Making It Expo-Ready

### Quick Demo Option (If Training Time is Limited)

1. **Use Transfer Learning**: Fine-tune only the last few layers
2. **Smaller Datasets**: Use 500-1000 images per disease (instead of 5000)
3. **Pre-trained Weights**: Use ImageNet weights, fine-tune minimally
4. **Focus on ONE Disease**: Perfect one module (e.g., Pneumonia) for live demo
5. **Show Training Process**: Display training curves and metrics

### Presentation Strategy

1. **Live Demo**: Show the working Streamlit interface
2. **Model Performance**: Display accuracy metrics, confusion matrices
3. **Unique Features**: Emphasize 5 color blindness tests and multi-modal fusion
4. **Technical Depth**: Explain CNN architectures, fusion methods, cross-validation
5. **PDF Reports**: Generate sample diagnostic reports during demo

## ⚡ Quick Start (Minimum Viable Demo)

If you have limited time, focus on:

1. **Pneumonia X-ray only**: Train ResNet50 on pneumonia dataset
2. **Use existing public datasets**: Download from Kaggle (ready to use)
3. **Skip audio, focus on images**: Images are easier and faster
4. **2-3 diseases maximum**: Pneumonia, Skin, Color blindness
5. **Demo mode with disclaimers**: Keep current demo mode but add clear "DEMO MODE" indicators

## 📌 Files to Update After Training

1. `models/pneumonia_model.py` - Replace random logic with model.predict()
2. `models/skin_model.py` - Load trained skin classifier
3. `models/heart_model.py` - Load saved Random Forest model
4. `models/audio_model.py` - Load audio CNN weights
5. `models/colorblind_model.py` - Load 5 trained test models

## 🎓 Academic Integrity Note

For your expo submission:
- Clearly cite all dataset sources
- Acknowledge use of pre-trained models (ResNet50, etc.)
- Explain the 5-dataset training methodology
- Show original contributions (fusion methods, 5 tests integration, PDF reports)

---

**The architecture is already built. You just need to add the trained weights!**

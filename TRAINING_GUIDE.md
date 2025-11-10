# Training Guide - Converting Demo to Production Models

## ⚠️ IMPORTANT: Current Implementation Status

The current application runs in **DEMO MODE** with simulated predictions. To make this expo-ready with real AI models, you must:

1. Collect real medical datasets (as outlined below)
2. Train the models using the provided architectures
3. Save trained model weights
4. Update model loading to use trained weights instead of demo mode

## 📊 Required Datasets

### 1. Pneumonia Detection (X-Ray Images)
**Dataset Sources:**
- Kaggle: "Chest X-Ray Images (Pneumonia)" dataset
- NIH Chest X-Ray Dataset
- RSNA Pneumonia Detection Challenge

**Requirements:**
- Minimum 5,000 X-ray images
- Classes: Normal vs Pneumonia
- Split into 5 separate folders (Dataset1, Dataset2, Dataset3, Dataset4, Dataset5)
- Each dataset: ~1,000 images

### 2. Pneumonia Audio (Cough/Breathing Sounds)
**Dataset Sources:**
- FluSense COVID-19 cough dataset
- ESC-50 Environmental Sound Classification
- Custom recorded cough sounds

**Requirements:**
- Minimum 2,000 audio files (.wav or .mp3)
- Labels: Normal breathing vs Abnormal (Pneumonia indicators)

### 3. Skin Disease Images
**Dataset Sources:**
- HAM10000 (dermatoscopic images)
- ISIC Archive (melanoma detection)
- DermNet skin disease dataset

**Requirements:**
- 7 classes: Acne, Eczema, Melanoma, Psoriasis, Dermatitis, Rosacea, Normal
- Minimum 500 images per class
- Split into 5 datasets

### 4. Heart Disease Clinical Data
**Dataset Sources:**
- UCI Heart Disease dataset
- Cleveland Heart Disease dataset
- Framingham Heart Study data

**Requirements:**
- Tabular data with features: age, sex, BP, cholesterol, heart rate, etc.
- Binary classification: Disease vs No Disease
- Minimum 5,000 patient records

### 5. Color Blindness Test Images
**Dataset Sources:**
- Create synthetic Ishihara plates
- Generate Farnsworth D-15 arrangement images
- Cambridge color test simulations

**Requirements:**
- 5 test types, each with multiple variations
- Labels for different CVD types (Protanopia, Deuteranopia, Tritanopia, Normal)

## 🔧 Training Steps

### Step 1: Organize Your Data
```bash
data/
├── pneumonia_xray/
│   ├── dataset1/
│   │   ├── normal/
│   │   └── pneumonia/
│   ├── dataset2/
│   ├── dataset3/
│   ├── dataset4/
│   └── dataset5/
├── pneumonia_audio/
│   ├── normal/
│   └── abnormal/
├── skin_disease/
│   ├── dataset1/
│   │   ├── acne/
│   │   ├── eczema/
│   │   └── ...
│   └── ...
├── heart_disease/
│   └── heart_data.csv
└── colorblind/
    ├── ishihara/
    ├── farnsworth/
    └── ...
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

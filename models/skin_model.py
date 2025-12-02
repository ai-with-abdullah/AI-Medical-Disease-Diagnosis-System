import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from PIL import Image
import cv2
import os

SKIN_CANCER_DEMO_CLASSES = [
    'Melanoma', 'Melanocytic Nevus (Mole)', 'Basal Cell Carcinoma',
    'Benign Keratosis', 'Actinic Keratosis', 'Vascular Lesion', 'Dermatofibroma'
]

# HAM10000 dataset classes mapping
HAM10000_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
HAM10000_NAMES = {
    'nv': 'Melanocytic Nevus (Mole)',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
}

TRAINED_MODEL_LOADED = False
TRAINED_MODEL = None

def load_trained_model():
    """Load trained skin cancer model if it exists"""
    global TRAINED_MODEL_LOADED, TRAINED_MODEL
    
    if TRAINED_MODEL_LOADED:
        return TRAINED_MODEL
    
    model_path = 'models/weights/skin_resnet50.h5'
    
    if os.path.exists(model_path):
        try:
            TRAINED_MODEL = keras.models.load_model(model_path)
            print(f"Loaded trained skin cancer model from {model_path}")
        except Exception as e:
            print(f"Error loading skin model: {e}. Using demo mode.")
            TRAINED_MODEL = None
    else:
        print(f"Skin model weights not found at {model_path}. Using demo mode.")
        TRAINED_MODEL = None
    
    TRAINED_MODEL_LOADED = True
    return TRAINED_MODEL

def preprocess_skin_image(image_pil, target_size=(224, 224)):
    """Preprocess skin image for model input"""
    img_array = np.array(image_pil.convert('RGB'))
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def analyze_skin_image(image_pil, model_choice):
    """Analyze skin image for cancer detection"""
    img_preprocessed = preprocess_skin_image(image_pil)
    
    # Load trained model
    model = load_trained_model()
    
    if model_choice == "Ensemble":
        # For ensemble, we'd load multiple models
        # Currently using single best model
        disease, confidence = get_single_skin_prediction(img_preprocessed, 'ResNet50')
        
        return {
            'disease': disease,
            'confidence': confidence,
            'category': get_disease_category(disease),
            'recommendations': get_recommendations(disease)
        }
    else:
        disease, confidence = get_single_skin_prediction(img_preprocessed, model_choice)
        return {
            'disease': disease,
            'confidence': confidence,
            'category': get_disease_category(disease),
            'recommendations': get_recommendations(disease)
        }

def get_single_skin_prediction(img_preprocessed, model_name):
    """Get prediction from skin cancer model"""
    
    if TRAINED_MODEL is not None:
        # PRODUCTION MODE: Use trained model
        prediction_probs = TRAINED_MODEL.predict(img_preprocessed, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction_probs[0])
        confidence = float(prediction_probs[0][predicted_class_idx])
        
        # Map to disease name based on training dataset
        if len(prediction_probs[0]) == 7 and len(HAM10000_CLASSES) == 7:
            # HAM10000 dataset classes
            disease_code = HAM10000_CLASSES[predicted_class_idx]
            disease = HAM10000_NAMES.get(disease_code, disease_code)
        else:
            # Custom 7-class model
            disease = SKIN_CANCER_DEMO_CLASSES[predicted_class_idx] if predicted_class_idx < len(SKIN_CANCER_DEMO_CLASSES) else 'Unknown'
        
        return disease, confidence
    
    else:
        # DEMO MODE: Use simulated predictions for skin cancer
        disease_idx = np.random.randint(0, len(SKIN_CANCER_DEMO_CLASSES))
        disease = SKIN_CANCER_DEMO_CLASSES[disease_idx]
        
        base_confidence = np.random.uniform(0.70, 0.95)
        confidence = base_confidence
        
        return disease, confidence

def get_disease_category(disease):
    """Get skin cancer category for classification"""
    categories = {
        'Melanoma': 'Malignant - High Risk',
        'Melanocytic Nevus (Mole)': 'Benign',
        'Benign Keratosis': 'Benign',
        'Basal Cell Carcinoma': 'Malignant - Medium Risk',
        'Actinic Keratosis': 'Pre-cancerous',
        'Vascular Lesion': 'Benign - Vascular',
        'Dermatofibroma': 'Benign'
    }
    return categories.get(disease, 'Unknown')

def get_recommendations(disease):
    """Get treatment recommendations for detected skin cancer condition"""
    recommendations = {
        'Melanoma': 'URGENT: Consult an oncologist immediately for biopsy and treatment options. Early detection is crucial for melanoma treatment success.',
        'Melanocytic Nevus (Mole)': 'Monitor for changes in size, shape, or color (ABCDE rule). Consult dermatologist if any changes occur.',
        'Benign Keratosis': 'Generally harmless seborrheic keratosis. Consult dermatologist if irritation occurs or for cosmetic removal.',
        'Basal Cell Carcinoma': 'IMPORTANT: Consult dermatologist or oncologist for treatment. BCC is usually treatable with early detection.',
        'Actinic Keratosis': 'Pre-cancerous condition requiring treatment. Consult dermatologist promptly to prevent progression to squamous cell carcinoma.',
        'Vascular Lesion': 'Usually benign vascular abnormality. Consult dermatologist for evaluation and treatment options if cosmetically concerning.',
        'Dermatofibroma': 'Benign fibrous growth. Usually no treatment needed unless causing discomfort or cosmetic concerns.'
    }
    return recommendations.get(disease, 'Consult a dermatologist or oncologist for proper diagnosis and treatment of this skin condition.')

def build_skin_classifier(base_model_name='resnet50', num_classes=7):
    """Build skin cancer classification model architecture"""
    if base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model for transfer learning
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def is_using_trained_model():
    """Check if system is using trained model or demo mode"""
    load_trained_model()
    return TRAINED_MODEL is not None

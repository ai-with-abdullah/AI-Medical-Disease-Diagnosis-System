import numpy as np
import os
import streamlit as st

@st.cache_resource
def get_tensorflow():
    """Lazy load TensorFlow - cached so only loads once"""
    import tensorflow as tf
    from tensorflow import keras
    return tf, keras

@st.cache_resource
def get_keras_applications():
    """Lazy load Keras applications - cached so only loads once"""
    from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
    from tensorflow.keras.preprocessing import image as keras_image
    return ResNet50, EfficientNetB0, MobileNetV2, keras_image

@st.cache_resource
def get_cv2():
    """Lazy load OpenCV - cached so only loads once"""
    import cv2
    return cv2

@st.cache_resource
def load_trained_model(model_name: str, model_path: str):
    """Load a single trained model - cached per model"""
    if not os.path.exists(model_path):
        return None
    
    try:
        _, keras = get_tensorflow()
        model = keras.models.load_model(model_path)
        print(f"✅ Loaded trained {model_name} model from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Error loading {model_name}: {e}. Using demo mode.")
        return None

def get_model_paths():
    """Get paths for all trained models"""
    return {
        'ResNet50': 'models/weights/pneumonia_resnet50.h5',
        'EfficientNet': 'models/weights/pneumonia_efficientnet.h5',
        'MobileNet': 'models/weights/pneumonia_mobilenet.h5'
    }

def get_trained_model(model_name: str):
    """Get a trained model by name (uses caching)"""
    model_paths = get_model_paths()
    if model_name not in model_paths:
        return None
    return load_trained_model(model_name, model_paths[model_name])

def preprocess_xray(image_pil, target_size=(224, 224)):
    """Preprocess X-ray image for model input"""
    from PIL import Image
    cv2 = get_cv2()
    
    img_array = np.array(image_pil.convert('RGB'))
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def analyze_xray_image(image_pil, model_choice):
    """Analyze chest X-ray image for pneumonia detection"""
    img_preprocessed = preprocess_xray(image_pil)
    
    if model_choice == "Ensemble (All Models)":
        predictions = []
        confidences = []
        model_names = []
        
        for model_name in ['ResNet50', 'EfficientNet', 'MobileNet']:
            pred, conf = get_single_model_prediction(img_preprocessed, model_name)
            predictions.append(pred)
            confidences.append(conf)
            model_names.append(model_name)
        
        normal_count = sum([1 for p in predictions if p == 'Normal'])
        pneumonia_count = sum([1 for p in predictions if p == 'Pneumonia'])
        
        if pneumonia_count > normal_count:
            final_pred = 'Pneumonia'
        else:
            final_pred = 'Normal'
        
        avg_confidence = np.mean(confidences)
        
        model_breakdown = [
            {'Model': name, 'Prediction': pred, 'Confidence': f"{conf:.2%}"}
            for name, pred, conf in zip(model_names, predictions, confidences)
        ]
        
        return {
            'prediction': final_pred,
            'confidence': avg_confidence,
            'model_used': 'Ensemble (ResNet50 + EfficientNet + MobileNet)',
            'model_breakdown': model_breakdown
        }
    else:
        prediction, confidence = get_single_model_prediction(img_preprocessed, model_choice)
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model_used': model_choice
        }

def get_single_model_prediction(img_preprocessed, model_name):
    """Get prediction from a single model (trained or demo)"""
    
    model = get_trained_model(model_name)
    
    if model is not None:
        prediction_probs = model.predict(img_preprocessed, verbose=0)
        
        if len(prediction_probs[0]) == 2:
            if prediction_probs[0][1] > 0.5:
                prediction = 'Pneumonia'
                confidence = float(prediction_probs[0][1])
            else:
                prediction = 'Normal'
                confidence = float(prediction_probs[0][0])
        else:
            pred_class = np.argmax(prediction_probs[0])
            confidence = float(prediction_probs[0][pred_class])
            prediction = 'Pneumonia' if pred_class == 1 else 'Normal'
        
        return prediction, confidence
    
    else:
        feature_score = np.random.random()
        
        if feature_score > 0.5:
            prediction = 'Pneumonia'
            confidence = 0.70 + (feature_score - 0.5) * 0.6
        else:
            prediction = 'Normal'
            confidence = 0.70 + (0.5 - feature_score) * 0.6
        
        noise = np.random.uniform(-0.05, 0.05)
        confidence = np.clip(confidence + noise, 0.6, 0.98)
        
        return prediction, confidence

@st.cache_resource
def build_pneumonia_classifier(base_model_name='resnet50', num_classes=2):
    """Build pneumonia classification model architecture"""
    _, keras = get_tensorflow()
    ResNet50, EfficientNetB0, MobileNetV2, _ = get_keras_applications()
    
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
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def is_using_trained_models():
    """Check if system is using trained models or demo mode"""
    model_paths = get_model_paths()
    trained_count = 0
    for model_name, path in model_paths.items():
        if get_trained_model(model_name) is not None:
            trained_count += 1
    return trained_count > 0, trained_count, len(model_paths)

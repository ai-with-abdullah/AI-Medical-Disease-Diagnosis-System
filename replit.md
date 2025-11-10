# AI Multi-Modal Disease Detection System

## Overview

This is a comprehensive medical diagnostic platform that combines Computer Vision, Deep Learning, Natural Language Processing, and Audio Processing to detect diseases across multiple modalities. The system currently operates in demo mode with simulated predictions and requires real dataset training to become production-ready.

The platform supports four main diagnostic categories:
1. **Pneumonia Detection** - Uses chest X-ray images and audio analysis (cough/breathing sounds)
2. **Skin Disease Detection** - Classifies 7 skin conditions from images
3. **Heart Disease Prediction** - Analyzes clinical parameters for risk assessment
4. **Color Blindness Detection** - Implements 5 comprehensive eye tests

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with custom CSS styling
- **Layout**: Wide layout with expandable sidebar navigation
- **UI Components**: Disease-specific cards with confidence-based color coding (green for high, yellow for medium, red for low confidence)
- **Entry Point**: `app.py` serves as the main Streamlit application

### Deep Learning Architecture
- **Transfer Learning**: Uses pre-trained ImageNet models (ResNet50, EfficientNetB0, MobileNetV2) as feature extractors
- **Ensemble Strategy**: Implements voting-based ensemble where multiple models predict and results are aggregated
- **Model Loading**: Currently configured for demo mode - models load pre-trained weights but use simulated predictions instead of real inference
- **Image Processing Pipeline**: PIL for image loading → NumPy array conversion → OpenCV resizing → normalization (0-1 range) → batch expansion

### Multi-Modal Fusion Engine
Located in `utils/fusion_engine.py`, this is a core architectural component that combines predictions from different data sources:
- **Fusion Methods**: Supports 4 different fusion strategies (Weighted Average, Voting Ensemble, Bayesian Inference, Stacking)
- **Modality Integration**: Combines image analysis, audio features, and text report analysis into a single diagnosis
- **Confidence Weighting**: Each modality contributes a prediction and confidence score that influences the final result

### Audio Processing Pipeline
- **Feature Extraction**: Uses librosa library to extract MFCC (40 coefficients), spectral centroid, spectral rolloff, zero-crossing rate, and chroma features
- **Visualization**: Generates MFCC plots and spectrograms for diagnostic insight
- **Audio Format**: Loads .wav or .mp3 files at 22050 Hz sample rate with 10-second duration

### NLP & OCR Processing
- **Text Extraction**: Uses pytesseract for OCR on PDF medical reports
- **PDF Processing**: Converts PDF pages to images using pdf2image before OCR
- **Medical Entity Extraction**: Regex-based parsing to extract vital signs, lab results, diagnoses, medications from unstructured text
- **Sentiment Analysis**: Analyzes medical report text to assess risk level (Concerning vs Normal)

### Training Architecture
The `training/train_models.py` implements a rigorous 5-dataset cross-validation strategy:
- **Dataset Split**: First 3 datasets for training, last 2 for testing
- **Validation**: 20% validation split during training
- **Metrics Tracking**: Records accuracy, precision, recall, F1-score, and confusion matrices
- **Training History**: Maintains JSON logs of all training phases and results

This approach is designed to demonstrate proper ML validation methodology but is not yet implemented with real data.

### Report Generation
- **Library**: ReportLab for PDF generation
- **Content**: Includes diagnosis summary, confidence scores, treatment recommendations, and visualizations
- **Format**: Professional clinical-grade layout with custom styling and color-coded results

### Data Models
The system expects specific data structures:
- **Pneumonia**: X-ray images (224x224 RGB) + audio files (10-second .wav/.mp3)
- **Skin Disease**: Dermoscopic images (224x224 RGB) across 7 classes
- **Heart Disease**: 9 clinical parameters (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Color Blindness**: Test images + user responses for 5 different test types

### Key Design Decisions

**Why Transfer Learning?**
- Leverages pre-trained ImageNet models to avoid training from scratch
- Reduces training time and data requirements
- Provides strong feature extraction for medical images

**Why Ensemble Methods?**
- Improves prediction reliability by combining multiple model perspectives
- Reduces individual model bias and variance
- Critical for medical applications where accuracy is paramount

**Why Multi-Modal Fusion?**
- Real medical diagnosis uses multiple data sources (images, symptoms, test results)
- Combining modalities increases diagnostic accuracy beyond single-source analysis
- Unique differentiator - no other student projects implement this level of integration

**Why Demo Mode?**
- Allows application demonstration without requiring large medical datasets
- Real medical datasets require ethics approval and are difficult to obtain
- Production deployment requires training on datasets outlined in TRAINING_GUIDE.md

## External Dependencies

### Core ML/AI Libraries
- **TensorFlow/Keras**: Deep learning framework for CNN models and transfer learning
- **scikit-learn**: Machine learning utilities (RandomForest, StandardScaler, train-test splits, metrics)
- **NumPy**: Numerical computing for array operations
- **OpenCV (cv2)**: Computer vision library for image preprocessing

### Audio Processing
- **librosa**: Audio analysis and feature extraction (MFCC, spectrograms, chroma features)
- **matplotlib**: Visualization of audio features and spectrograms

### NLP & Document Processing
- **pytesseract**: OCR engine for extracting text from images
- **pdf2image**: Converts PDF pages to PIL images for OCR processing
- **Pillow (PIL)**: Image loading, manipulation, and format conversion

### Web Application
- **Streamlit**: Web framework for building the interactive UI
- **pandas**: Data manipulation and tabular data handling

### Report Generation
- **ReportLab**: PDF generation library for creating diagnostic reports

### Image Processing
- **PIL/Pillow**: Primary image handling library
- **OpenCV**: Advanced image processing and transformations

### Expected External Services
While not currently integrated, production deployment would benefit from:
- **Cloud Storage**: For storing trained model weights (AWS S3, Google Cloud Storage)
- **Database**: For patient records and diagnosis history (not currently implemented)
- **Authentication Service**: For patient/doctor login (not currently implemented)

### Dataset Sources (for Training)
As outlined in TRAINING_GUIDE.md, the following external datasets are required:
- **Kaggle**: Chest X-Ray Images (Pneumonia dataset)
- **NIH**: Chest X-Ray Dataset
- **HAM10000**: Dermatoscopic images
- **ISIC Archive**: Melanoma detection dataset
- **UCI**: Heart Disease dataset
- **FluSense**: COVID-19 cough dataset

### System Requirements
- **Python**: 3.8+
- **Tesseract OCR**: Must be installed separately on the system for pytesseract to work
- **Poppler**: Required for pdf2image PDF rendering
- **FFmpeg**: May be required for certain audio format conversions in librosa
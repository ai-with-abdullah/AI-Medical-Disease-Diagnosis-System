# AI Multi-Modal Disease Detection System

## 🏥 Project Overview

An advanced medical diagnostic platform combining **Computer Vision**, **Deep Learning**, **NLP**, and **Audio Processing** for comprehensive disease detection across multiple modalities.

## 🎯 Key Features

### Supported Diseases
1. **Pneumonia Detection**
   - Chest X-ray analysis using ResNet50, EfficientNet, MobileNet
   - Audio analysis of cough and breathing sounds
   - Multi-modal fusion for enhanced accuracy

2. **Skin Disease Detection**
   - Image-based classification of 7 skin conditions
   - Ensemble CNN models
   - Treatment recommendations

3. **Heart Disease Prediction**
   - Clinical parameter analysis
   - Random Forest classifier
   - Risk assessment and feature importance

4. **Color Blindness Detection**
   - 5 comprehensive eye tests:
     - Ishihara Plates Test
     - Farnsworth D-15 Test
     - Cambridge Color Test
     - Color Spectrum Discrimination
     - Anomaloscope Simulation
   - Ensemble analysis for accurate diagnosis

### Advanced Capabilities
- **Multi-Modal Fusion**: Combines image, audio, and text report analysis
- **PDF Report Generation**: Professional diagnostic reports with visualizations
- **Multiple Fusion Methods**: Weighted averaging, voting ensemble, Bayesian inference, stacking
- **Model Performance Tracking**: Cross-validation results and metrics visualization

## 🔬 Technologies Used

### Deep Learning & ML
- TensorFlow/Keras for neural networks
- Pre-trained models: ResNet50, EfficientNet, MobileNet
- Custom CNN architectures
- Random Forest for tabular data
- Ensemble methods

### Computer Vision
- OpenCV for image processing
- PIL for image handling
- Medical image preprocessing

### Audio Processing
- Librosa for audio feature extraction
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral analysis
- Cough and breathing pattern recognition

### NLP & OCR
- PyTesseract for text extraction from PDFs
- Medical report text analysis
- Named entity recognition for clinical data

### Data Science
- NumPy, Pandas for data manipulation
- Scikit-learn for ML utilities
- Matplotlib, Seaborn for visualization

### Web Framework
- Streamlit for interactive web interface
- Real-time predictions
- Multi-page navigation

## 📊 Training Strategy

### 5-Dataset Cross-Validation Approach
1. **Phase 1**: Train models on 3 datasets (60% of data)
2. **Phase 2**: Validate on remaining 2 datasets (40% of data)
3. **Phase 3**: Fine-tune based on validation results
4. **Phase 4**: Retrain on all 5 datasets for final model
5. **Phase 5**: Cross-validation for robust performance estimation

This methodology ensures:
- Robust model generalization
- Reduced overfitting
- Realistic performance metrics
- Multiple validation checkpoints

## 🚀 Getting Started

### Prerequisites
All dependencies are pre-installed:
- Python 3.11
- TensorFlow
- Scikit-learn
- Librosa
- OpenCV
- PyTesseract
- Streamlit
- ReportLab (PDF generation)

### Running the Application
The application is configured to run automatically. Access it through the web preview.

### Project Structure
```
.
├── app.py                      # Main Streamlit application
├── models/                     # ML model implementations
│   ├── pneumonia_model.py      # Pneumonia detection models
│   ├── skin_model.py           # Skin disease classification
│   ├── heart_model.py          # Heart disease prediction
│   ├── audio_model.py          # Audio processing for pneumonia
│   └── colorblind_model.py     # 5 color blindness tests
├── utils/                      # Utility functions
│   ├── nlp_processor.py        # Medical report NLP & OCR
│   ├── fusion_engine.py        # Multi-modal fusion algorithms
│   └── pdf_generator.py        # PDF report generation
├── training/                   # Training scripts
│   └── train_models.py         # 5-dataset training pipeline
└── assets/                     # Sample data and resources
```

## 🎨 Features in Detail

### 1. Pneumonia Detection
- **Image Analysis**: Upload chest X-rays for CNN-based analysis
- **Audio Analysis**: Analyze cough/breathing sounds using MFCC features
- **Multi-Modal**: Combine both for higher accuracy
- **Models**: ResNet50, EfficientNet, MobileNet ensemble

### 2. Skin Disease Detection
- Detects: Acne, Eczema, Melanoma, Psoriasis, Dermatitis, Rosacea
- Provides treatment recommendations
- Ensemble model voting for robustness

### 3. Heart Disease Prediction
- Clinical parameters: Age, BP, Cholesterol, Heart Rate, etc.
- Random Forest classifier
- Feature importance visualization
- Risk level assessment (High/Medium/Low)

### 4. Color Blindness Tests
- **Ishihara Plates**: Classic red-green deficiency detection
- **Farnsworth D-15**: Color arrangement and sequencing
- **Cambridge Test**: Pattern detection in chromatic contrasts
- **Spectrum Discrimination**: Gradient-based color matching
- **Anomaloscope**: Gold-standard clinical simulation
- **Ensemble Analysis**: Combines all 5 tests for final diagnosis

### 5. Multi-Modal Fusion
- Upload multiple inputs (image + audio + report)
- 4 fusion methods:
  - Weighted Average
  - Voting Ensemble
  - Bayesian Inference
  - Stacking
- Confidence score aggregation
- Professional PDF report generation

## 📄 PDF Report Features
- Comprehensive diagnosis summary
- Individual modality results
- Confidence scores and visualizations
- Clinical recommendations
- Professional medical formatting
- Timestamp and metadata

## 🏆 Why This Project Stands Out

1. **Multi-Modal Analysis**: Unique combination of image, audio, and text
2. **5 Color Blindness Tests**: Most comprehensive eye testing suite
3. **Advanced Fusion**: Multiple ensemble methods
4. **Professional Reports**: Clinical-grade PDF generation
5. **Rigorous Training**: 5-dataset cross-validation strategy
6. **Real-World Impact**: Practical healthcare application
7. **Technical Depth**: Demonstrates CV, NLP, Audio, ML, DL

## 📚 For Training Your Models

To train models with your own datasets:

1. Collect medical datasets for each disease category
2. Organize into 5 separate dataset folders
3. Preprocess images (resize, normalize, augment)
4. Run training pipeline: `python training/train_models.py`
5. Follow the 5-dataset strategy outlined in the code
6. Save trained models for deployment

## ⚠️ Important Disclaimer

This system is designed for **educational and research purposes only**. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## 🤝 Team Information

- **Team Size**: 2-3 members
- **Development Time**: 1 month
- **Target**: Expo presentation
- **Focus**: AI/ML, Computer Vision, NLP, Data Science

## 📞 Support

For questions about the implementation, refer to the code documentation within each module.

---

**Built with ❤️ using TensorFlow, Streamlit, and Advanced AI/ML Techniques**
# AI-Medical-Disease-Diagnosis-System

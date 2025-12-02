# Deployment & Local Setup Guide - AI Medical Diagnosis App

This guide explains how to run your app locally on your laptop and deploy it online.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Local Setup (Your Laptop)](#local-setup-your-laptop)
   - [Prerequisites](#prerequisites)
   - [Step-by-Step Installation](#step-by-step-installation)
   - [Running the App](#running-the-app)
   - [Training Models Locally](#training-models-locally)
3. [Troubleshooting](#troubleshooting)
4. [Online Deployment (Replit)](#online-deployment-replit)

---

## Project Structure

```
/
├── app.py                      # Main Streamlit application
├── assets/                     # Color blindness test images (30 images)
├── models/                     # AI model definitions
│   ├── audio_model.py          # Pneumonia audio analysis
│   ├── colorblind_model.py     # Color blindness testing
│   ├── heart_model.py          # Heart disease prediction
│   ├── pneumonia_model.py      # X-ray analysis
│   └── skin_model.py           # Skin lesion classification
├── models/weights/             # Trained model weights (after training)
├── utils/                      # Utility functions
│   ├── fusion_engine.py        # Multi-modal fusion
│   ├── nlp_processor.py        # Text processing
│   └── pdf_generator.py        # PDF report generation
├── training/                   # Model training code
├── training_scripts/           # Individual training scripts
│   ├── train_pneumonia_models.py       # Train X-ray models
│   ├── train_pneumonia_audio_models.py # Train audio models
│   ├── train_skin_model.py             # Train skin cancer model
│   ├── train_heart_models.py           # Train heart disease models
│   └── preprocess_pneumonia_data.py    # Data preprocessing
├── training_data/              # Place your datasets here
│   ├── pneumonia/              # X-ray datasets
│   ├── pneumonia_audio/        # Audio datasets
│   ├── skin_cancer/            # Skin cancer datasets
│   ├── heart_disease/          # Heart disease datasets
│   └── arrhythmia/             # Arrhythmia datasets
├── requirements.txt            # Python dependencies
├── COMPREHENSIVE_TRAINING_GUIDE.md  # How to train all models
└── README.md                   # Project documentation
```

---

## Local Setup (Your Laptop)

### Prerequisites

Before starting, make sure you have:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.9, 3.10, or 3.11 (recommended: 3.11) | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |

**Important:** Use Python 3.11 for best compatibility. Avoid Python 3.12+ as some libraries may not be fully compatible yet.

---

### Step-by-Step Installation

#### Step 1: Clone or Download the Project

```bash
# Option A: Clone with Git
git clone <your-repository-url>
cd <project-folder>

# Option B: Download ZIP and extract
# Extract to a folder, then open terminal in that folder
cd /path/to/extracted/folder
```

#### Step 2: Create a Virtual Environment

**Why use a virtual environment?**
- Prevents conflicts with other Python projects
- Isolates library versions
- Ensures reproducibility

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) at the start of your terminal prompt
```

**On Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat

# You should see (venv) at the start of your terminal prompt
```

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 3: Upgrade pip (Important!)

```bash
# Upgrade pip to latest version
pip install --upgrade pip
```

#### Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Expected installation time:** 5-15 minutes (depending on internet speed)

**If you encounter errors:**
```bash
# Try installing packages one by one
pip install streamlit
pip install tensorflow
pip install opencv-python
pip install numpy pandas pillow
pip install scikit-learn scipy matplotlib
pip install librosa
pip install reportlab pdf2image
pip install seaborn
```

#### Step 5: Verify Installation

```bash
# Check if key packages are installed
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import librosa; print('Librosa:', librosa.__version__)"
```

---

### Running the App

#### Start the Application

```bash
# Make sure virtual environment is activated!
# You should see (venv) in your terminal

# Run the Streamlit app
streamlit run app.py --server.port 5000
```

#### Access the App

After running, you'll see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:5000
Network URL: http://192.168.x.x:5000
```

**Open your browser and go to:** `http://localhost:5000`

#### Stop the App

Press `Ctrl + C` in the terminal to stop the server.

#### Deactivate Virtual Environment (When Done)

```bash
deactivate
```

---

### Training Models Locally

Before training, make sure you've downloaded the datasets as described in `COMPREHENSIVE_TRAINING_GUIDE.md`.

#### Train All Models (One by One)

```bash
# Activate virtual environment first!
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows

# 1. Train Pneumonia X-Ray Models (3 CNN models)
python training_scripts/train_pneumonia_models.py

# 2. Train Pneumonia Audio Models (Random Forest + Neural Network)
python training_scripts/train_pneumonia_audio_models.py

# 3. Train Skin Cancer Model (ResNet50)
python training_scripts/train_skin_model.py

# 4. Train Heart Disease Models (3 Random Forest models)
python training_scripts/train_heart_models.py
```

#### Training Time Estimates

| Model | Dataset Size | Training Time | Expected Accuracy |
|-------|-------------|---------------|-------------------|
| Pneumonia X-Ray (3 models) | 5,000-100,000 images | 1-8 hours | 90-97% |
| Pneumonia Audio (2 models) | 5,000-30,000 recordings | 10 min - 2 hours | 85-95% |
| Skin Cancer (1 model) | 10,000-50,000 images | 30 min - 3 hours | 88-95% |
| Heart Disease (3 models) | 70,000-600,000 records | 2-15 minutes | 85-95% |

#### Trained Model Locations

After training, models are saved to `models/weights/`:

```
models/weights/
├── pneumonia_resnet50.h5          # Pneumonia X-ray (ResNet50)
├── pneumonia_efficientnet.h5      # Pneumonia X-ray (EfficientNet)
├── pneumonia_mobilenet.h5         # Pneumonia X-ray (MobileNet)
├── pneumonia_audio_rf_model.pkl   # Pneumonia Audio (Random Forest)
├── pneumonia_audio_rf_scaler.pkl  # Audio RF Scaler
├── pneumonia_audio_nn_model.h5    # Pneumonia Audio (Neural Network)
├── pneumonia_audio_nn_scaler.pkl  # Audio NN Scaler
├── skin_cancer_model.h5           # Skin Cancer (ResNet50)
├── heart_generic_model.pkl        # Heart Disease (Generic CVD)
├── heart_generic_scaler.pkl       # Heart Generic Scaler
├── heart_cad_model.pkl            # Heart Disease (CAD)
├── heart_cad_scaler.pkl           # Heart CAD Scaler
├── heart_arrhythmia_model.pkl     # Heart Disease (Arrhythmia)
└── heart_arrhythmia_scaler.pkl    # Heart Arrhythmia Scaler
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'xxx'"

**Solution:** Install the missing package
```bash
pip install <package-name>
```

#### Issue 2: "Python version mismatch"

**Solution:** Create a new virtual environment with Python 3.11
```bash
# macOS/Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

#### Issue 3: TensorFlow installation fails

**Solution:** Try installing a specific version
```bash
pip install tensorflow==2.15.0
```

#### Issue 4: librosa installation fails (audio processing)

**Solution:** Install system dependencies first

**macOS:**
```bash
brew install libsndfile
pip install librosa
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libsndfile1
pip install librosa
```

**Windows:**
```bash
pip install soundfile
pip install librosa
```

#### Issue 5: OpenCV installation fails

**Solution:** Use headless version
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

#### Issue 6: Port 5000 already in use

**Solution:** Use a different port
```bash
streamlit run app.py --server.port 8080
```

#### Issue 7: Virtual environment not activating

**Solution:** Check if you're in the right directory
```bash
# List files to verify venv folder exists
ls -la   # macOS/Linux
dir      # Windows

# If venv doesn't exist, create it again
python -m venv venv
```

### Checking Your Setup

Run this command to verify everything is working:
```bash
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import streamlit; print(f'Streamlit: OK ({streamlit.__version__})')
except: print('Streamlit: MISSING')
try:
    import tensorflow; print(f'TensorFlow: OK ({tensorflow.__version__})')
except: print('TensorFlow: MISSING')
try:
    import sklearn; print(f'Scikit-learn: OK ({sklearn.__version__})')
except: print('Scikit-learn: MISSING')
try:
    import librosa; print(f'Librosa: OK ({librosa.__version__})')
except: print('Librosa: MISSING')
try:
    import cv2; print(f'OpenCV: OK ({cv2.__version__})')
except: print('OpenCV: MISSING')
try:
    import numpy; print(f'NumPy: OK ({numpy.__version__})')
except: print('NumPy: MISSING')
try:
    import pandas; print(f'Pandas: OK ({pandas.__version__})')
except: print('Pandas: MISSING')
"
```

---

## Online Deployment (Replit)

If you want to deploy the app online and make it accessible to others:

### Deployment Type

Select **"Autoscale"** for this web application:
- **Autoscale**: Best for web apps - automatically scales based on traffic
- **Reserved VM**: For apps that need to run continuously
- **Static**: For simple HTML/CSS/JS websites only

### Run Command

The run command is already configured:
```bash
streamlit run app.py --server.port 5000
```

### After Deployment

Your app will be accessible at a public URL like:
```
https://your-app-name.replit.app
```

---

## Quick Reference Commands

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py --server.port 5000

# Deactivate
deactivate

# Train models
python training_scripts/train_pneumonia_models.py
python training_scripts/train_pneumonia_audio_models.py
python training_scripts/train_skin_model.py
python training_scripts/train_heart_models.py
```

---

## Team Members

- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz

---

**Need help?** Check the `COMPREHENSIVE_TRAINING_GUIDE.md` for detailed dataset and training instructions.

# How to Deploy Your AI Medical Diagnosis App

This guide explains how to make your app run automatically and be accessible to others.

## Current Project Structure (Clean)

```
/
├── app.py                 # Main Streamlit application
├── assets/                # Color blindness test images
├── models/                # AI model definitions
│   ├── audio_model.py
│   ├── colorblind_model.py
│   ├── heart_model.py
│   ├── pneumonia_model.py
│   └── skin_model.py
├── utils/                 # Utility functions
│   ├── fusion_engine.py
│   ├── nlp_processor.py
│   └── pdf_generator.py
├── training/              # Model training code
├── training_scripts/      # Individual training scripts
├── training_data/         # Place your datasets here
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```


### Choose Deployment Type (if you want to deploy the app online and want to accessible with the others)
Select "Autoscale" for a web application like this one:
- **Autoscale**: Best for web apps - automatically scales based on traffic
- **Reserved VM**: For apps that need to run continuously
- **Static**: For simple HTML/CSS/JS websites only

### Configure Build & Run Commands
These are already set up for you:
- **Run command**: `streamlit run app.py --server.port 9000 --server.address 0.0.0.0 --server.headless true`
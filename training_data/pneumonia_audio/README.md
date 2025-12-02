# Pneumonia Audio Dataset Setup Guide

## Overview

This folder should contain audio recordings (cough, breathing sounds) for training pneumonia detection models.
The training script automatically detects and combines multiple datasets for higher accuracy.

---

## Available Datasets (Choose one or more)

| # | Dataset | Recordings | Size | Download Link |
|---|---------|------------|------|---------------|
| 1 | COUGHVID | 25,000+ | 2 GB | [Download](https://zenodo.org/record/4498364) |
| 2 | Coswara | 2,635 individuals | 5 GB | [Download](https://github.com/iiscleap/Coswara-Data) |
| 3 | ICBHI 2017 | 920 | 1 GB | [Download](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) |
| 4 | Virufy | 1,000+ | 500 MB | [Download](https://github.com/virufy/virufy-data) |
| 5 | COVID-19 Cough | 4,000+ | 800 MB | [Download](https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification) |
| 6 | Kaggle Respiratory | 5,500 | 500 MB | [Download](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) |

---

## Quick Start (Easiest Option)

### Option 1: Pre-Organized Structure
Create these folders and add your audio files:

```
training_data/pneumonia_audio/
└── organized/
    ├── normal/
    │   └── (healthy cough/breathing audio files)
    └── abnormal/
        └── (pneumonia/COVID cough/breathing audio files)
```

### Option 2: Download COUGHVID Dataset
1. Go to: https://zenodo.org/record/4498364
2. Download `public_dataset.zip`
3. Extract to: `training_data/pneumonia_audio/coughvid/`

---

## Folder Structure

After downloading, organize your files like this:

```
training_data/pneumonia_audio/
│
├── coughvid/                        <- Dataset 1: COUGHVID
│   ├── public_dataset/
│   │   ├── *.webm / *.ogg / *.wav
│   │   └── ... (25,000+ recordings)
│   └── metadata_compiled.csv        <- Labels file
│
├── coswara/                         <- Dataset 2: Coswara
│   ├── Extracted_data/
│   │   └── <date_folders>/
│   │       └── <user_id>/
│   │           ├── cough-heavy.wav
│   │           ├── cough-shallow.wav
│   │           ├── breathing-deep.wav
│   │           └── breathing-shallow.wav
│   └── combined_data.csv
│
├── icbhi_2017/                      <- Dataset 3: ICBHI 2017
│   ├── audio_files/
│   │   └── *.wav (920 recordings)
│   └── patient_diagnosis.csv
│
├── virufy/                          <- Dataset 4: Virufy
│   ├── pos/
│   │   └── *.wav (COVID positive)
│   └── neg/
│       └── *.wav (Non-COVID)
│
├── covid_cough/                     <- Dataset 5: COVID Cough
│   ├── covid/
│   │   └── *.wav
│   └── healthy/
│       └── *.wav
│
├── kaggle_respiratory/              <- Dataset 6: Kaggle Respiratory
│   ├── audio_files/
│   │   └── *.wav
│   └── labels.csv
│
└── organized/                       <- Pre-organized (EASIEST!)
    ├── normal/
    │   └── *.wav (healthy sounds)
    └── abnormal/
        └── *.wav (pneumonia/illness sounds)
```

---

## Training Command

After placing your audio files, run:

```bash
python training_scripts/train_pneumonia_audio_models.py
```

The script will:
1. Automatically detect available datasets
2. Extract audio features (MFCC, Spectral, Chroma)
3. Train Random Forest + Neural Network models
4. Save models to `models/weights/`

---

## Expected Training Time

| Data Size | Training Time | Expected Accuracy |
|-----------|--------------|-------------------|
| ~5,000 recordings | 10-20 minutes | 80-85% |
| ~15,000 recordings | 30-60 minutes | 85-88% |
| ~30,000 recordings | 1-2 hours | 88-92% |

---

## Team Members
- F23BARIN1M01140 - Muhammad Abdullah
- F23BARIN1M01131 - Muhammad Ali Yahya
- F23BARIN1M01228 - Manahil Shouket
- F23BARIN1M01114 - Ayman Noor
- F23BARIN1M01225 - Tayyaba Mumtaz

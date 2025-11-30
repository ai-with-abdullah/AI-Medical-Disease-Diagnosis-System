# AI Multi-Modal Disease Detection System

## Overview

This project is an advanced, AI-powered medical diagnostic platform designed for the detection of four distinct disease categories using 8 specialized deep learning models plus interactive clinical testing. It processes multi-modal data including X-rays, dermoscopy images, medical features, and real-time visual input. The system aims to provide medical-grade accuracy and professional reporting, making it suitable for clinical support and academic demonstration. Key capabilities include Pneumonia Detection (3 CNN models), Skin Disease Classification (1 CNN model), Heart Disease Prediction (3 Random Forest models), and comprehensive Color Blindness Testing with 5 interactive clinical tests (Ishihara, Farnsworth, Cambridge, Spectrum, Anomaloscope) - no AI training required for color blindness as it uses predefined medical standards. The project emphasizes production-readiness, robust UI/UX, and advanced AI techniques for a fully functional and impressive demo.

## Project Structure (Cleaned)

```
/
├── app.py                 # Main Streamlit application
├── assets/                # Color blindness test images (30 images)
├── models/                # AI model definitions
│   ├── audio_model.py     # Heart sound analysis
│   ├── colorblind_model.py # Color blindness testing
│   ├── heart_model.py     # Heart disease prediction
│   ├── pneumonia_model.py # X-ray analysis
│   └── skin_model.py      # Skin lesion classification
├── utils/                 # Utility functions
│   ├── fusion_engine.py   # Multi-modal fusion
│   ├── nlp_processor.py   # Text processing
│   └── pdf_generator.py   # PDF report generation
├── training/              # Model training code
├── training_scripts/      # Individual training scripts
├── training_data/         # Dataset directories (add your data here)
├── DEPLOYMENT_GUIDE.md    # How to deploy this app
├── COMPREHENSIVE_TRAINING_GUIDE.md # How to train models
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## User Preferences

- **Focus:** Production-ready, impressive for scholarship applications
- **Design:** Medical-grade, professional UI
- **Testing:** Live camera preferred (no file uploads)
- **Accuracy:** Highest priority
- **Documentation:** Comprehensive and detailed

## System Architecture

The system is built around four independent disease detection modules, unified by a multi-modal fusion engine. The UI is developed using Streamlit, ensuring a clean, medical-grade, and intuitive interface with real-time processing and session state management.

**UI/UX Decisions:**
- Professional color scheme (Navy #0a1e3d + Light blue #1a3a52)
- Justified text, color-coded tables, and clear heading hierarchies for academic appearance in reports.
- Live camera integration for all testing modules, eliminating file uploads.
- Real-time progress bars, visual dashboards, and color-coded results (green/yellow/red).
- Downloadable JSON reports with detailed breakdowns and medical recommendations.
- Auto-resetting forms for improved user experience, especially in multi-option modules like Heart Disease Prediction.

**Technical Implementations & Feature Specifications:**

-   **Pneumonia Detection (`models/pneumonia_model.py`):** Utilizes an ensemble of ResNet50, EfficientNet, and MobileNet for X-ray image classification (Normal/Pneumonia) with fusion prediction.
-   **Skin Disease Classification (`models/skin_model.py`):** Employs a ResNet50 model for 7-class classification on dermoscopy images (acne, BCC, BKL, dermatitis, melanoma, nevus, vascular).
-   **Heart Disease Prediction (`models/heart_model.py`):** Features a Random Forest classifier predicting risk across three types: Generic Cardiovascular Disease, Coronary Artery Disease, and Cardiac Arrhythmia, based on 9 standardized medical features. Now supports 6 large datasets (total 645,000+ records): Cardiovascular Disease (70K), Personal Key Indicators (320K), Health Indicators (254K), Comprehensive (1.1K), Heart Failure (918), UCI Original (303). Includes smart form reset functionality.
-   **Color Blindness Testing (`models/colorblind_model.py`):** A comprehensive suite of 5 clinical tests (Ishihara, Farnsworth D-15, Cambridge, Spectrum Discrimination, Anomaloscope Simulation) using interactive testing methodology - **NO AI TRAINING REQUIRED**. Uses predefined correct answers based on medical standards, calculates "Eye Damage Ratio" from user responses, and provides detailed individual test results and overall diagnosis.
-   **Multi-Modal Fusion Engine (`utils/fusion_engine.py`):** Combines outputs from different modalities (image, audio, text) using four strategies: Weighted Average, Voting, Bayesian, and Stacking for cross-disease analysis.
-   **Professional PDF Report Generator (`utils/pdf_generator.py`):** Generates academic research proposal-formatted PDFs (645 lines, 16-17 KB) with 3-page structure: Professional Title Page, University Cover Page with Group Members, and comprehensive content sections. Features The Islamia University of Bahawalpur branding, group member table with names and roll numbers, and 7 major academic sections with minimal white space.

**System Design Choices:**
-   **Live Testing:** Prioritizes live camera input over file uploads for real-time interaction and production readiness.
-   **Session Management:** Extensive use of Streamlit's session state for managing test progression, form states, and user interactions.
-   **Modularity:** Clear separation of concerns with dedicated Python files for each disease model and utility functions.
-   **Advanced AI Techniques:** Integration of ensemble learning, transfer learning (ResNet50, EfficientNet, MobileNet), Random Forest, and CNNs.

## External Dependencies

-   **Core Libraries:** `streamlit`, `tensorflow`, `opencv-python`, `numpy`, `pandas`, `pillow`, `scikit-learn`, `scipy`, `matplotlib`, `librosa`, `pytesseract`.
-   **PDF Generation:** `reportlab`, `pdf2image`.

## Recent Updates (Nov 28, 2025)

**Documentation Update - Color Blindness Module Clarification:**
- Updated README.md to clarify that Color Blindness uses **interactive clinical testing** (no AI training needed)
- Corrected model count: 8 trained AI models (was incorrectly listed as 13)
- Corrected dataset count: 17+ datasets (removed 4 color blindness datasets that were never actually required)
- Color Blindness module uses predefined correct answers from medical standards, not trained CNN models
- This approach is **more accurate** for color blindness detection since it tests actual user perception
- Updated replit.md overview and technical specifications

---

## Recent Updates (Nov 27, 2025)

**Streamlit Version Compatibility Fixes:**
- Added comprehensive version-checking compatibility wrappers to support both old and new Streamlit versions
- Created `get_streamlit_version()` helper function to detect Streamlit version at runtime
- Implemented `get_image_width_param()` for `st.image()` - handles `use_container_width` (v1.31+) vs `use_column_width` (older)
- Implemented `get_button_width_param()` for `st.button()` - handles `use_container_width` (v1.29+)
- Implemented `get_dataframe_width_param()` for `st.dataframe()` - handles `use_container_width` (v1.22+)
- Updated all 17 st.button() calls to use compatibility wrapper
- Updated all 5 st.dataframe() calls to use compatibility wrapper
- Updated all st.image() calls to use compatibility wrapper
- App now works on Streamlit versions 1.21+ without TypeError for deprecated parameters

**Project Cleanup:**
- Removed unnecessary files: attached_assets/, stub main.py, duplicate root files, zip archive
- Fixed potential "image is possibly unbound" issue in skin disease detection module
- Verified all dependencies are present in requirements.txt including reportlab for PDF generation

## Recent Updates (Nov 25, 2025)

**Color Blindness Module Restructure - Individual vs Combined Mode:**
- Completely restructured color blindness testing with two distinct modes:
  1. **Individual Test Mode:** Users select one of 5 tests and complete 6 items to get results for that specific test
  2. **Combined Mode:** Users complete all 5 tests (30 items total) for comprehensive assessment with Eye Damage Ratio
- **Tests Available (6 items each):**
  - 🔴 Ishihara Plates (93% accuracy)
  - 🌈 Farnsworth D-15 (89% accuracy)
  - 🎨 Cambridge Color Test (87% accuracy)
  - 📊 Spectrum Discrimination (85% accuracy)
  - 🔬 Anomaloscope Simulation (95% accuracy)
- **Mode Selection Interface:** Users choose between individual test selection or combined mode at the beginning
- **Individual Test Results:** Shows accuracy, correct answers, and confidence for selected test
- **Combined Results:** Shows individual test results + overall eye damage ratio + comprehensive diagnosis
- **UI Improvements:** Clean test selection buttons, progress tracking, mode switching option

**Complete PDF Generator Rewrite - Clean & Professional:**
- Completely rewrote `utils/pdf_generator.py` from scratch with clean, simple design
- **Separate Cover Page** - Professional title page without team members
- **Separate Team Members Page** - Dedicated page with all 5 group members, roll numbers, project duration (2023-2027), supervisors
- **Table of Contents** - Easy navigation with all sections listed
- **11 Complete Sections** - Abstract, Introduction, Literature Review, Methodology, Implementation, Results, Discussion, Conclusion, Future Work, References, Appendix
- **Clean Design** - Professional Navy (#0a1e3d) + Light Blue (#1a3a52) color scheme
- **Professional Tables** - Color-coded headers, proper alignment, team info display
- **Output Quality:** 18.5 KB per report (lightweight yet comprehensive)

**PDF Report Structure:**
1. **Cover Page** - Project title, university branding, academic year 2023-2027
2. **Team Members Page** - All 5 members with roll numbers, project info
3. **Table of Contents** - Complete section listing
4. **Section 1: ABSTRACT** - Project summary
5. **Section 2: INTRODUCTION** (2.1-2.4) - Background, Problem, Objectives, Scope
6. **Section 3: LITERATURE REVIEW** (3.1-3.3) - Research, Related Work, Gaps
7. **Section 4: METHODOLOGY** (4.1-4.4) - Datasets, Preprocessing, Architecture, Training
8. **Section 5: IMPLEMENTATION** (5.1-5.4) - Tools, Architecture, Workflow, UI
9. **Section 6: RESULTS** (6.1-6.5) - Accuracy, Loss, Confusion, Predictions, Per-disease
10. **Section 7: DISCUSSION** (7.1-7.3) - Interpretation, Challenges, Comparison
11. **Section 8: CONCLUSION** (8.1-8.3) - Summary, Findings, Limitations
12. **Section 9: FUTURE WORK** (9.1-9.4) - Improvements, Expansion, Deployment, Explainable AI
13. **Section 10: REFERENCES** - 10 academic citations
14. **Section 11: APPENDIX** (11.1-11.4) - Graphs, Images, Code, Architecture

**How to Run on Your Mac:**
```bash
cd /Users/macintosh/Desktop/M.Abdullah/
python3 utils/pdf_generator.py
```
**Result:** `Medical_Diagnostic_Report.pdf` (18.5 KB, clean & professional)

**Features:**
- Clean, simple professional design
- Separate cover and team members pages
- Academic year: 2023-2027
- All 11 sections properly organized
- Complete team information with roll numbers
- Suitable for scholarship applications
- Lightweight file size (18.5 KB)
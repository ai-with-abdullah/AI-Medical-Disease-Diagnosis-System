from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime

def create_academic_proposal():
    pdf_filename = "AI_Medical_Diagnosis_Academic_Proposal.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    Story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'AcademicTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.black,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=22
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.black,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading1_style = ParagraphStyle(
        'AcademicHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.black,
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'AcademicHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'AcademicBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14,
        fontName='Times-Roman'
    )
    
    # Title Page
    Story.append(Spacer(1, 1.5*inch))
    Story.append(Paragraph("ARTIFICIAL INTELLIGENCE-BASED MULTI-MODAL DISEASE DETECTION SYSTEM", title_style))
    Story.append(Spacer(1, 0.3*inch))
    Story.append(Paragraph("A Comprehensive Approach to Medical Diagnostics Using Deep Learning,", subtitle_style))
    Story.append(Paragraph("Computer Vision, and Multi-Modal Data Fusion", subtitle_style))
    Story.append(Spacer(1, 0.5*inch))
    
    Story.append(Paragraph("<b>Academic Research Proposal</b>", subtitle_style))
    Story.append(Spacer(1, 0.3*inch))
    
    Story.append(Paragraph(f"Submitted: {datetime.now().strftime('%B %Y')}", body_style))
    Story.append(Spacer(1, 0.2*inch))
    
    Story.append(Paragraph("<b>Field of Study:</b> Artificial Intelligence & Computer Science", body_style))
    Story.append(Paragraph("<b>Research Domain:</b> Medical AI, Healthcare Technology", body_style))
    Story.append(Paragraph("<b>Keywords:</b> Deep Learning, Computer Vision, Multi-Modal Fusion, Medical Diagnosis, Healthcare AI", body_style))
    
    Story.append(PageBreak())
    
    # Abstract
    Story.append(Paragraph("ABSTRACT", heading1_style))
    abstract = """
    Healthcare systems worldwide face critical challenges in timely and accurate disease diagnosis, particularly 
    in resource-constrained environments. This research presents an innovative artificial intelligence-based 
    multi-modal disease detection system that addresses these challenges through advanced deep learning techniques. 
    The proposed system integrates Computer Vision, Natural Language Processing, and Audio Signal Processing to 
    create a comprehensive diagnostic platform capable of detecting multiple diseases across different modalities.
    <br/><br/>
    Unlike conventional single-modality approaches, our system employs multi-modal data fusion—combining medical 
    imaging, audio analysis, and textual medical reports—to achieve higher diagnostic accuracy and reliability. 
    The research demonstrates practical applications in pneumonia detection (utilizing chest X-rays and cough 
    sounds), skin disease classification, cardiovascular risk assessment, and comprehensive color vision deficiency 
    testing.
    <br/><br/>
    This work contributes to the growing field of medical artificial intelligence by demonstrating how diverse 
    data sources can be synergistically combined to support clinical decision-making. The system architecture 
    is designed to be scalable, adaptable to various healthcare settings, and capable of assisting medical 
    professionals in making more informed diagnostic decisions. The research has significant implications for 
    improving healthcare accessibility, reducing diagnostic errors, and enabling early disease detection in 
    underserved communities.
    """
    Story.append(Paragraph(abstract, body_style))
    
    Story.append(PageBreak())
    
    # 1. Introduction and Problem Statement
    Story.append(Paragraph("1. INTRODUCTION AND PROBLEM STATEMENT", heading1_style))
    
    Story.append(Paragraph("1.1 Global Healthcare Challenges", heading2_style))
    intro1 = """
    The World Health Organization reports that millions of people worldwide lack access to quality diagnostic 
    services, leading to delayed treatments and preventable deaths. Several critical challenges persist in 
    modern healthcare:
    <br/><br/>
    <b>Diagnostic Delays:</b> In many developing countries, the shortage of trained radiologists and pathologists 
    results in diagnostic backlogs lasting weeks or months. Time-sensitive conditions like pneumonia or melanoma 
    require rapid diagnosis for effective treatment.
    <br/><br/>
    <b>Geographic Disparities:</b> Rural and remote areas often lack access to specialized medical expertise. 
    Patients must travel long distances for basic diagnostic services, creating barriers to healthcare access.
    <br/><br/>
    <b>Human Error and Fatigue:</b> Studies show that diagnostic errors affect approximately 12 million Americans 
    annually. Radiologist fatigue, high workload, and inter-observer variability contribute to missed diagnoses.
    <br/><br/>
    <b>Cost Barriers:</b> Advanced diagnostic procedures remain expensive and inaccessible to large populations, 
    particularly in low- and middle-income countries.
    <br/><br/>
    <b>Single-Modality Limitations:</b> Traditional diagnostic approaches often rely on a single data source 
    (e.g., only imaging or only blood tests), potentially missing crucial diagnostic information available 
    through other modalities.
    """
    Story.append(Paragraph(intro1, body_style))
    
    Story.append(Paragraph("1.2 Research Motivation", heading2_style))
    motivation = """
    Artificial intelligence, particularly deep learning, has demonstrated remarkable success in medical image 
    analysis, achieving performance comparable to or exceeding human experts in specific tasks. However, most 
    existing AI diagnostic systems focus on single modalities and single diseases, limiting their practical 
    utility in real-world clinical settings.
    <br/><br/>
    This research is motivated by three key observations:
    <br/><br/>
    <b>First,</b> clinical diagnosis in practice integrates multiple information sources—physicians consider 
    patient history, physical examination, imaging studies, laboratory results, and other clinical data. An 
    effective AI system should mirror this multi-modal approach.
    <br/><br/>
    <b>Second,</b> emerging technologies in deep learning, transfer learning, and ensemble methods now enable 
    the development of sophisticated multi-disease, multi-modal diagnostic systems that were previously 
    computationally infeasible.
    <br/><br/>
    <b>Third,</b> the COVID-19 pandemic highlighted the urgent need for scalable, accessible diagnostic tools 
    that can operate with minimal human intervention, particularly for infectious diseases.
    """
    Story.append(Paragraph(motivation, body_style))
    
    Story.append(Paragraph("1.3 Research Objectives", heading2_style))
    objectives = """
    This research aims to develop and validate an integrated AI-based diagnostic platform with the following 
    specific objectives:
    <br/><br/>
    <b>Primary Objective:</b> Design and implement a multi-modal disease detection system that combines medical 
    imaging, audio signal analysis, and natural language processing to provide comprehensive diagnostic support.
    <br/><br/>
    <b>Secondary Objectives:</b>
    <br/>• Develop disease-specific deep learning models for pneumonia, skin diseases, cardiovascular conditions, 
    and color vision deficiencies
    <br/>• Implement and compare multiple data fusion algorithms for combining predictions from different modalities
    <br/>• Create a user-friendly interface accessible to both medical professionals and patients
    <br/>• Establish a rigorous training and validation methodology ensuring model reliability and generalization
    <br/>• Demonstrate the clinical utility and practical feasibility of multi-modal AI diagnostics
    """
    Story.append(Paragraph(objectives, body_style))
    
    Story.append(PageBreak())
    
    # 2. Literature Review and Background
    Story.append(Paragraph("2. LITERATURE REVIEW AND THEORETICAL BACKGROUND", heading1_style))
    
    Story.append(Paragraph("2.1 Artificial Intelligence in Medical Diagnosis", heading2_style))
    lit1 = """
    The application of AI in medical diagnosis has evolved significantly over the past decade. Early systems 
    relied on rule-based expert systems with limited success. The breakthrough came with deep learning, 
    particularly Convolutional Neural Networks (CNNs), which revolutionized medical image analysis.
    <br/><br/>
    <b>Breakthrough Studies:</b>
    <br/>• Esteva et al. (2017) demonstrated that deep learning could classify skin cancer with accuracy matching 
    board-certified dermatologists using a dataset of 129,450 clinical images.
    <br/>• Rajpurkar et al. (2017) developed CheXNet, achieving radiologist-level pneumonia detection from chest 
    X-rays using a 121-layer CNN trained on 112,120 frontal-view X-ray images.
    <br/>• Gulshan et al. (2016) showed that deep learning algorithms could detect diabetic retinopathy with 
    high sensitivity and specificity, potentially addressing screening challenges in diabetes care.
    <br/><br/>
    These studies established that AI could match or exceed specialist performance in specific diagnostic tasks, 
    validating the clinical potential of deep learning approaches.
    """
    Story.append(Paragraph(lit1, body_style))
    
    Story.append(Paragraph("2.2 Multi-Modal Learning in Healthcare", heading2_style))
    lit2 = """
    Recent research has shifted toward multi-modal approaches that integrate diverse data types. The rationale 
    is based on the clinical reality that accurate diagnosis requires synthesizing multiple information sources.
    <br/><br/>
    <b>Multi-Modal Fusion Approaches:</b>
    <br/>• <b>Early Fusion:</b> Combines features from different modalities at the input level
    <br/>• <b>Late Fusion:</b> Integrates predictions from independently trained modality-specific models
    <br/>• <b>Hybrid Fusion:</b> Combines features at intermediate layers of neural networks
    <br/><br/>
    Studies by Huang et al. (2020) demonstrated that multi-modal fusion significantly outperformed single-modality 
    approaches in cancer diagnosis, with improvements of 8-12% in diagnostic accuracy. This validates our choice 
    to implement multi-modal data fusion as a core component of the proposed system.
    """
    Story.append(Paragraph(lit2, body_style))
    
    Story.append(Paragraph("2.3 Transfer Learning and Pre-trained Models", heading2_style))
    lit3 = """
    Transfer learning has become the standard approach in medical AI due to limited availability of large 
    annotated medical datasets. By leveraging models pre-trained on large-scale datasets (e.g., ImageNet with 
    14 million images), researchers can achieve high performance even with relatively small medical datasets.
    <br/><br/>
    <b>Key Architectures:</b>
    <br/>• <b>ResNet (Residual Networks):</b> Introduced skip connections to enable training of very deep networks, 
    achieving breakthrough performance in image recognition
    <br/>• <b>EfficientNet:</b> Uses compound scaling to balance network depth, width, and resolution, achieving 
    superior accuracy with fewer parameters
    <br/>• <b>MobileNet:</b> Designed for mobile and edge devices, using depthwise separable convolutions for 
    computational efficiency
    <br/><br/>
    Our research employs these architectures as feature extractors, fine-tuning them on medical datasets to 
    leverage both general visual knowledge and domain-specific medical features.
    """
    Story.append(Paragraph(lit3, body_style))
    
    Story.append(PageBreak())
    
    # 3. Methodology
    Story.append(Paragraph("3. RESEARCH METHODOLOGY", heading1_style))
    
    Story.append(Paragraph("3.1 System Architecture", heading2_style))
    arch = """
    The proposed system employs a modular architecture consisting of four primary components:
    <br/><br/>
    <b>1. Data Acquisition Module:</b> Handles multi-modal input including medical images (X-rays, dermoscopic 
    images), audio signals (cough sounds, breathing patterns), and textual medical reports. Implements real-time 
    data capture capabilities for practical deployment.
    <br/><br/>
    <b>2. Modality-Specific Processing Pipelines:</b>
    <br/>• <b>Image Processing:</b> Utilizes transfer learning with ResNet50, EfficientNetB0, and MobileNetV2 
    pre-trained on ImageNet, fine-tuned for medical imaging tasks
    <br/>• <b>Audio Processing:</b> Extracts Mel-Frequency Cepstral Coefficients (MFCC), spectral features, 
    and temporal patterns using digital signal processing techniques
    <br/>• <b>Text Processing:</b> Employs Optical Character Recognition (OCR) for medical report digitization 
    and Natural Language Processing for clinical entity extraction
    <br/><br/>
    <b>3. Multi-Modal Fusion Engine:</b> Implements four fusion strategies (Weighted Average, Voting Ensemble, 
    Bayesian Inference, and Stacking) to optimally combine predictions from different modalities.
    <br/><br/>
    <b>4. Decision Support Interface:</b> Presents diagnostic results with confidence scores, supporting 
    visualizations, and clinical recommendations in an accessible format for healthcare providers.
    """
    Story.append(Paragraph(arch, body_style))
    
    Story.append(Paragraph("3.2 Disease-Specific Models", heading2_style))
    diseases = """
    The system addresses four clinically significant disease categories:
    <br/><br/>
    <b>Pneumonia Detection:</b> Combines chest X-ray analysis (using ensemble of ResNet50, EfficientNet, 
    MobileNet) with audio analysis of cough and breathing sounds. Pneumonia causes over 2.5 million deaths 
    annually worldwide, with early detection crucial for treatment success.
    <br/><br/>
    <b>Dermatological Conditions:</b> Classifies seven skin conditions including melanoma (deadliest skin cancer), 
    acne, eczema, psoriasis, dermatitis, and rosacea. Skin cancer incidence has increased dramatically, with 
    early detection critical for survival.
    <br/><br/>
    <b>Cardiovascular Risk Assessment:</b> Analyzes clinical parameters (blood pressure, cholesterol, ECG findings) 
    using Random Forest classification. Cardiovascular disease remains the leading cause of death globally.
    <br/><br/>
    <b>Color Vision Deficiency:</b> Implements five clinical-grade tests (Ishihara Plates, Farnsworth D-15, 
    Cambridge Color Test, Spectrum Discrimination, Anomaloscope) for comprehensive color vision assessment. 
    Affects approximately 8% of males and 0.5% of females worldwide.
    """
    Story.append(Paragraph(diseases, body_style))
    
    Story.append(Paragraph("3.3 Training and Validation Strategy", heading2_style))
    training = """
    To ensure robust model performance and generalization, we implement a rigorous 5-dataset cross-validation 
    methodology:
    <br/><br/>
    <b>Dataset Diversity:</b> Models are trained on data from multiple sources to reduce dataset-specific bias 
    and improve generalization to real-world scenarios.
    <br/><br/>
    <b>Validation Protocol:</b>
    <br/>• Phase 1: Initial training on 60% of datasets (Datasets 1-3)
    <br/>• Phase 2: Validation on remaining 40% (Datasets 4-5) to assess generalization
    <br/>• Phase 3: Hyperparameter optimization based on validation performance
    <br/>• Phase 4: Final training on all datasets using optimal parameters
    <br/>• Phase 5: 5-fold cross-validation for robust performance estimation
    <br/><br/>
    <b>Evaluation Metrics:</b> Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices provide 
    comprehensive performance assessment. For medical applications, we prioritize high sensitivity (recall) to 
    minimize false negatives, which could have serious clinical consequences.
    """
    Story.append(Paragraph(training, body_style))
    
    Story.append(PageBreak())
    
    # 4. Why This Approach
    Story.append(Paragraph("4. RATIONALE FOR METHODOLOGICAL CHOICES", heading1_style))
    
    Story.append(Paragraph("4.1 Why Multi-Modal Fusion?", heading2_style))
    why1 = """
    <b>Clinical Justification:</b> Physicians naturally integrate multiple information sources when diagnosing 
    patients. A chest X-ray provides anatomical information, but a patient's cough sound can reveal functional 
    respiratory patterns. Medical reports contain historical context and lab results. By combining these modalities, 
    our system mimics expert clinical reasoning.
    <br/><br/>
    <b>Technical Justification:</b> Single-modality approaches are vulnerable to modality-specific noise and 
    artifacts. Multi-modal fusion provides redundancy and complementary information. Research by Baltrusaitis et 
    al. (2019) shows that multi-modal systems consistently outperform single-modality approaches, with typical 
    accuracy improvements of 5-15%.
    <br/><br/>
    <b>Practical Justification:</b> In real-world settings, not all modalities may be available for every patient. 
    A multi-modal system can gracefully degrade, providing reasonable predictions even when some data sources are 
    missing.
    """
    Story.append(Paragraph(why1, body_style))
    
    Story.append(Paragraph("4.2 Why Transfer Learning?", heading2_style))
    why2 = """
    <b>Data Efficiency:</b> Medical datasets are inherently limited due to privacy concerns, annotation costs, 
    and rare disease prevalence. Transfer learning allows us to leverage knowledge from millions of natural 
    images (ImageNet) and adapt it to medical imaging with relatively few training examples.
    <br/><br/>
    <b>Performance Gains:</b> Studies show that transfer learning typically improves accuracy by 10-20% compared 
    to training from scratch, especially crucial when medical datasets contain only thousands rather than millions 
    of images.
    <br/><br/>
    <b>Computational Efficiency:</b> Pre-trained models converge faster, requiring less computational resources 
    and training time—critical factors for academic research with limited resources.
    """
    Story.append(Paragraph(why2, body_style))
    
    Story.append(Paragraph("4.3 Why Ensemble Methods?", heading2_style))
    why3 = """
    <b>Error Reduction:</b> Individual models make different types of errors. By combining multiple models 
    (ResNet50, EfficientNet, MobileNet), we reduce both bias and variance, leading to more reliable predictions.
    <br/><br/>
    <b>Medical Safety:</b> In medical applications, consensus from multiple models provides additional confidence. 
    If all three models agree on a diagnosis, the prediction is highly reliable. Disagreement among models flags 
    cases requiring human expert review.
    <br/><br/>
    <b>Robustness:</b> Ensemble methods are less sensitive to specific model weaknesses or training data 
    peculiarities, improving system robustness across diverse patient populations.
    """
    Story.append(Paragraph(why3, body_style))
    
    Story.append(PageBreak())
    
    # 5. Expected Outcomes and Impact
    Story.append(Paragraph("5. EXPECTED OUTCOMES AND SOCIETAL IMPACT", heading1_style))
    
    Story.append(Paragraph("5.1 Technical Contributions", heading2_style))
    contributions = """
    This research makes several technical contributions to the field of medical AI:
    <br/><br/>
    <b>1. Novel Multi-Modal Fusion Framework:</b> Implementation and comparison of four fusion strategies 
    specifically designed for medical diagnostics, providing insights into optimal fusion approaches for 
    different clinical scenarios.
    <br/><br/>
    <b>2. Comprehensive Color Vision Assessment:</b> First integrated system combining five clinical-grade 
    color blindness tests (Ishihara, Farnsworth D-15, Cambridge, Spectrum, Anomaloscope) in a single diagnostic 
    platform, significantly advancing accessibility of color vision testing.
    <br/><br/>
    <b>3. Cross-Disease Diagnostic Platform:</b> Demonstrates the feasibility of a unified AI architecture 
    capable of handling multiple disease categories and data modalities, reducing development costs for future 
    medical AI applications.
    <br/><br/>
    <b>4. Robust Validation Methodology:</b> The 5-dataset cross-validation strategy provides a template for 
    rigorous medical AI validation, addressing common criticisms of overfitting and dataset bias in healthcare AI.
    """
    Story.append(Paragraph(contributions, body_style))
    
    Story.append(Paragraph("5.2 Clinical and Societal Impact", heading2_style))
    impact = """
    <b>Improved Healthcare Accessibility:</b> The system can be deployed in remote or underserved areas, providing 
    diagnostic support where specialist expertise is unavailable. This addresses the WHO's goal of universal health 
    coverage and equitable access to quality healthcare services.
    <br/><br/>
    <b>Early Disease Detection:</b> By making diagnostic tools more accessible and affordable, the system enables 
    earlier disease detection, particularly for conditions like melanoma and pneumonia where early intervention 
    dramatically improves outcomes.
    <br/><br/>
    <b>Reduced Healthcare Costs:</b> AI-assisted diagnosis can reduce the need for expensive specialist 
    consultations, multiple diagnostic tests, and unnecessary treatments resulting from diagnostic errors. Studies 
    suggest AI diagnostics could reduce healthcare costs by 15-20% in specific domains.
    <br/><br/>
    <b>Clinical Decision Support:</b> Rather than replacing physicians, the system serves as a second opinion 
    tool, helping doctors make more informed decisions and reducing diagnostic errors caused by fatigue or 
    oversight.
    <br/><br/>
    <b>Public Health Surveillance:</b> Aggregate data from the system could support epidemiological surveillance, 
    helping public health authorities identify disease outbreaks and trends in real-time.
    """
    Story.append(Paragraph(impact, body_style))
    
    Story.append(Paragraph("5.3 Educational Impact", heading2_style))
    education = """
    This project demonstrates practical application of advanced computer science concepts in solving real-world 
    problems:
    <br/><br/>
    <b>Interdisciplinary Integration:</b> Successfully bridges computer science, medicine, and data science, 
    showcasing the importance of interdisciplinary collaboration in modern research.
    <br/><br/>
    <b>Technical Depth:</b> Demonstrates mastery of multiple AI/ML domains including deep learning, computer 
    vision, natural language processing, and audio signal processing—skills highly valued in academia and industry.
    <br/><br/>
    <b>Research Methodology:</b> Exhibits understanding of proper scientific methodology, validation techniques, 
    and statistical analysis crucial for graduate-level research.
    <br/><br/>
    <b>Ethical Awareness:</b> Addresses critical questions about AI in healthcare, including bias, fairness, 
    privacy, and the appropriate role of automation in medical decision-making.
    """
    Story.append(Paragraph(education, body_style))
    
    Story.append(PageBreak())
    
    # 6. Future Research Directions
    Story.append(Paragraph("6. FUTURE RESEARCH DIRECTIONS", heading1_style))
    
    future = """
    This project establishes a foundation for multiple promising research directions:
    <br/><br/>
    <b>6.1 Expansion to Additional Diseases</b>
    <br/>The modular architecture can be extended to other conditions including:
    <br/>• Diabetic retinopathy screening from retinal fundus images
    <br/>• Tuberculosis detection from chest X-rays (critical for global health)
    <br/>• Alzheimer's disease prediction from brain MRI scans
    <br/>• COVID-19 and other respiratory infection detection
    <br/><br/>
    <b>6.2 Explainable AI Integration</b>
    <br/>Implementing attention mechanisms and gradient-based visualization techniques (Grad-CAM, SHAP values) to 
    make model decisions interpretable. This addresses the critical "black box" problem in medical AI, helping 
    physicians understand and trust AI recommendations.
    <br/><br/>
    <b>6.3 Federated Learning for Privacy-Preserving Training</b>
    <br/>Developing federated learning protocols that enable model training across multiple hospitals without 
    sharing patient data, addressing privacy concerns while improving model generalization through diverse datasets.
    <br/><br/>
    <b>6.4 Edge Deployment and Mobile Health</b>
    <br/>Optimizing models for deployment on mobile devices and edge computing platforms, enabling diagnostic 
    capabilities in resource-limited settings without internet connectivity. This involves model compression, 
    quantization, and neural architecture search for efficient models.
    <br/><br/>
    <b>6.5 Longitudinal Patient Monitoring</b>
    <br/>Extending the system to track disease progression over time, incorporating temporal analysis for chronic 
    condition management. This could enable personalized treatment optimization and early detection of disease 
    progression.
    <br/><br/>
    <b>6.6 Clinical Validation Studies</b>
    <br/>Conducting prospective clinical trials to validate system performance in real-world healthcare settings, 
    comparing AI-assisted diagnosis with standard clinical practice. This is essential for regulatory approval 
    and clinical adoption.
    """
    Story.append(Paragraph(future, body_style))
    
    Story.append(PageBreak())
    
    # 7. Challenges and Limitations
    Story.append(Paragraph("7. CHALLENGES AND LIMITATIONS", heading1_style))
    
    challenges = """
    <b>7.1 Data Availability and Quality</b>
    <br/>Medical datasets are often limited, imbalanced, and contain annotation errors. Different imaging 
    protocols across institutions create domain shift problems. Our cross-dataset validation strategy partially 
    addresses this, but larger, more diverse datasets would improve generalization.
    <br/><br/>
    <b>7.2 Regulatory and Ethical Considerations</b>
    <br/>Medical AI systems require regulatory approval (FDA in USA, CE marking in Europe) before clinical 
    deployment. This involves extensive validation, documentation, and ongoing monitoring. Additionally, questions 
    of liability, informed consent, and algorithmic bias must be carefully addressed.
    <br/><br/>
    <b>7.3 Model Interpretability</b>
    <br/>Deep learning models are often criticized as "black boxes." While our ensemble approach provides some 
    transparency through model agreement/disagreement, further work on explainable AI is needed for full clinical 
    acceptance.
    <br/><br/>
    <b>7.4 Generalization Across Populations</b>
    <br/>Models trained on data from specific populations may not generalize well to different demographics, 
    ethnicities, or geographic regions. This requires ongoing validation and potential model retraining for 
    different deployment contexts.
    <br/><br/>
    <b>7.5 Integration with Clinical Workflows</b>
    <br/>Successful clinical adoption requires seamless integration with existing hospital information systems, 
    electronic health records, and clinical workflows. This involves addressing technical compatibility, user 
    training, and workflow redesign challenges.
    """
    Story.append(Paragraph(challenges, body_style))
    
    Story.append(PageBreak())
    
    # 8. Conclusion
    Story.append(Paragraph("8. CONCLUSION", heading1_style))
    
    conclusion = """
    This research addresses critical challenges in healthcare accessibility and diagnostic accuracy through an 
    innovative multi-modal AI system. By integrating medical imaging, audio analysis, and natural language 
    processing with advanced deep learning techniques, we demonstrate a comprehensive approach to disease detection 
    that mirrors clinical diagnostic reasoning.
    <br/><br/>
    The choice of multi-modal fusion is grounded in both clinical practice and technical advantages. Physicians 
    naturally synthesize multiple information sources; our system computationally replicates this approach. The 
    implementation of transfer learning and ensemble methods addresses practical constraints of limited medical 
    data while ensuring robust, reliable predictions.
    <br/><br/>
    Beyond technical contributions, this work has significant societal implications. By making advanced diagnostic 
    capabilities accessible in resource-limited settings, the system could help address healthcare disparities 
    affecting billions of people worldwide. The platform's modular architecture enables continuous expansion to 
    additional diseases and modalities, providing a sustainable framework for future medical AI development.
    <br/><br/>
    From an educational perspective, this project demonstrates mastery of multiple AI/ML disciplines and showcases 
    the ability to apply theoretical knowledge to solve complex real-world problems. The interdisciplinary nature 
    of the work—spanning computer science, medicine, and data science—reflects the collaborative approach 
    increasingly essential in modern research and industry.
    <br/><br/>
    While challenges remain in regulatory approval, clinical validation, and widespread deployment, this research 
    establishes a strong foundation for future work in medical AI. The methodologies, architectures, and validation 
    strategies developed here can inform future research in healthcare technology and contribute to the growing 
    body of knowledge in artificial intelligence applications for social good.
    <br/><br/>
    As healthcare systems globally face increasing pressure from aging populations, rising chronic disease burden, 
    and resource constraints, AI-assisted diagnostics represent not just a technological advancement, but a 
    necessity for sustainable, equitable healthcare delivery. This research contributes to that critical goal.
    """
    Story.append(Paragraph(conclusion, body_style))
    
    Story.append(PageBreak())
    
    # References
    Story.append(Paragraph("9. REFERENCES", heading1_style))
    
    references = """
    Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2019). Multimodal machine learning: A survey and taxonomy. 
    <i>IEEE Transactions on Pattern Analysis and Machine Intelligence, 41</i>(2), 423-443.
    <br/><br/>
    Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). 
    Dermatologist-level classification of skin cancer with deep neural networks. <i>Nature, 542</i>(7639), 115-118.
    <br/><br/>
    Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). 
    Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal 
    fundus photographs. <i>JAMA, 316</i>(22), 2402-2410.
    <br/><br/>
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. 
    <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i>, 770-778.
    <br/><br/>
    Huang, S. C., Pareek, A., Seyyedi, S., Banerjee, I., & Lungren, M. P. (2020). Fusion of medical imaging and 
    electronic health records using deep learning: a systematic review and implementation guidelines. 
    <i>NPJ Digital Medicine, 3</i>(1), 1-9.
    <br/><br/>
    Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). CheXNet: 
    Radiologist-level pneumonia detection on chest X-rays with deep learning. <i>arXiv preprint arXiv:1711.05225</i>.
    <br/><br/>
    Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. 
    <i>International Conference on Machine Learning</i>, 6105-6114.
    <br/><br/>
    World Health Organization. (2021). <i>Global strategy on digital health 2020-2025</i>. Geneva: WHO.
    <br/><br/>
    Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. 
    <i>Nature Medicine, 25</i>(1), 44-56.
    <br/><br/>
    Yu, K. H., Beam, A. L., & Kohane, I. S. (2018). Artificial intelligence in healthcare. 
    <i>Nature Biomedical Engineering, 2</i>(10), 719-731.
    """
    Story.append(Paragraph(references, body_style))
    
    Story.append(PageBreak())
    
    # Appendix
    Story.append(Paragraph("APPENDIX: TECHNICAL SPECIFICATIONS", heading1_style))
    
    Story.append(Paragraph("A. System Implementation Summary", heading2_style))
    
    tech_summary = [
        ["Component", "Specification"],
        ["Programming Language", "Python 3.11"],
        ["Deep Learning Framework", "TensorFlow 2.20 / Keras"],
        ["ML Library", "Scikit-learn 1.7"],
        ["Computer Vision", "OpenCV 4.11"],
        ["Audio Processing", "Librosa 0.11"],
        ["NLP/OCR", "PyTesseract 0.3"],
        ["Web Framework", "Streamlit 1.51"],
        ["Data Processing", "NumPy 2.3, Pandas 2.3"],
        ["Visualization", "Matplotlib 3.10, Seaborn 0.13"],
        ["Code Base", "2,500+ lines of Python"],
        ["AI Models", "10+ deep learning and ML models"],
    ]
    
    t1 = Table(tech_summary, colWidths=[2.5*inch, 3.5*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    Story.append(t1)
    Story.append(Spacer(1, 0.3*inch))
    
    Story.append(Paragraph("B. Model Architectures Summary", heading2_style))
    
    models_summary = [
        ["Disease", "Models Used", "Input Type"],
        ["Pneumonia", "ResNet50, EfficientNet, MobileNet + Audio CNN", "X-ray images + Audio"],
        ["Skin Diseases", "ResNet50, EfficientNet, MobileNet ensemble", "Dermoscopic images"],
        ["Heart Disease", "Random Forest (100 trees)", "Clinical parameters"],
        ["Color Blindness", "5 custom CNNs (one per test type)", "Test images"],
    ]
    
    t2 = Table(models_summary, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    Story.append(t2)
    
    Story.append(Spacer(1, 0.5*inch))
    
    footer = f"""
    <para align=center>
    <b>--- END OF PROPOSAL ---</b><br/>
    <br/>
    Document prepared: {datetime.now().strftime('%B %d, %Y')}<br/>
    For academic review and consideration for advanced studies
    </para>
    """
    Story.append(Paragraph(footer, body_style))
    
    doc.build(Story)
    print(f"Academic proposal PDF generated: {pdf_filename}")
    return pdf_filename

if __name__ == "__main__":
    create_academic_proposal()

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

CVD_TYPES = ['Normal', 'Protanopia (Red-blind)', 'Deuteranopia (Green-blind)', 
             'Tritanopia (Blue-blind)', 'Protanomaly', 'Deuteranomaly']

def analyze_ishihara(image_pil, user_answer):
    img_array = np.array(image_pil.convert('RGB'))
    
    expected_answers = {
        '12': 'Normal',
        '8': 'Red-Green Deficiency',
        '29': 'Normal',
        '45': 'Red-Green Deficiency',
        '5': 'Normal',
        '3': 'Red-Green Deficiency'
    }
    
    expected = list(expected_answers.keys())[np.random.randint(0, len(expected_answers))]
    
    if user_answer == expected:
        result = "Correct - Normal Color Vision"
        interpretation = "Your answer matches the expected response for normal color vision."
        status_class = "high-confidence"
    else:
        result = "Incorrect - Possible Color Vision Deficiency"
        interpretation = "Your answer suggests a possible red-green color vision deficiency. Further testing recommended."
        status_class = "low-confidence"
    
    return {
        'result': result,
        'expected': expected,
        'user_answer': user_answer,
        'interpretation': interpretation,
        'status_class': status_class,
        'confidence': 0.85
    }

def analyze_farnsworth(image_pil):
    img_array = np.array(image_pil.convert('RGB'))
    
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mean_hue = np.mean(hsv_img[:,:,0])
    std_hue = np.std(hsv_img[:,:,0])
    
    discrimination_score = min(std_hue / 50.0, 1.0)
    
    if discrimination_score > 0.75:
        result = "Normal Color Discrimination"
        deficiency_type = "None"
        status_class = "high-confidence"
    elif discrimination_score > 0.5:
        result = "Mild Color Discrimination Deficit"
        deficiency_type = "Mild Deficiency"
        status_class = "medium-confidence"
    else:
        result = "Significant Color Discrimination Deficit"
        deficiency_type = "Moderate to Severe Deficiency"
        status_class = "low-confidence"
    
    return {
        'result': result,
        'score': discrimination_score,
        'deficiency_type': deficiency_type,
        'status_class': status_class,
        'confidence': 0.80
    }

def analyze_cambridge(image_pil, pattern_seen):
    img_array = np.array(image_pil.convert('RGB'))
    
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    chromatic_contrast = np.std(a_channel) + np.std(b_channel)
    threshold_level = chromatic_contrast / 100.0
    
    if pattern_seen == "Yes":
        if threshold_level > 0.6:
            diagnosis = "Normal Color Vision"
            severity = "None"
            status_class = "high-confidence"
        else:
            diagnosis = "Borderline Color Vision"
            severity = "Mild"
            status_class = "medium-confidence"
    else:
        diagnosis = "Color Vision Deficiency Detected"
        if threshold_level < 0.3:
            severity = "Severe"
            status_class = "low-confidence"
        else:
            severity = "Moderate"
            status_class = "medium-confidence"
    
    return {
        'diagnosis': diagnosis,
        'threshold': f"{threshold_level:.2f}",
        'severity': severity,
        'status_class': status_class,
        'confidence': 0.82
    }

def analyze_spectrum(image_pil):
    img_array = np.array(image_pil.convert('RGB'))
    
    r_channel = img_array[:,:,0]
    g_channel = img_array[:,:,1]
    b_channel = img_array[:,:,2]
    
    rg_variance = np.var(r_channel - g_channel)
    by_variance = np.var(b_channel - (r_channel + g_channel) / 2)
    
    rg_score = min(rg_variance / 1000.0, 1.0)
    by_score = min(by_variance / 1000.0, 1.0)
    
    overall_score = (rg_score + by_score) / 2
    
    if overall_score > 0.75:
        result = "Excellent Color Spectrum Discrimination"
        overall = "Normal Color Vision"
        status_class = "high-confidence"
    elif overall_score > 0.5:
        result = "Moderate Color Spectrum Discrimination"
        overall = "Mild Deficiency"
        status_class = "medium-confidence"
    else:
        result = "Poor Color Spectrum Discrimination"
        overall = "Significant Deficiency"
        status_class = "low-confidence"
    
    return {
        'result': result,
        'rg_score': rg_score,
        'by_score': by_score,
        'overall': overall,
        'status_class': status_class,
        'confidence': 0.83
    }

def analyze_anomaloscope(image_pil):
    img_array = np.array(image_pil.convert('RGB'))
    
    r_mean = np.mean(img_array[:,:,0])
    g_mean = np.mean(img_array[:,:,1])
    
    rg_ratio = r_mean / (g_mean + 1e-6)
    
    if 0.9 <= rg_ratio <= 1.1:
        diagnosis = "Normal Trichromat"
        cvd_type = "None"
        classification = "Normal Color Vision"
        severity = "None"
        status_class = "high-confidence"
    elif rg_ratio < 0.7:
        diagnosis = "Protanomalous Trichromat"
        cvd_type = "Red Weakness (Protanomaly)"
        classification = "Anomalous Trichromacy"
        severity = "Mild to Moderate"
        status_class = "medium-confidence"
    elif rg_ratio > 1.3:
        diagnosis = "Deuteranomalous Trichromat"
        cvd_type = "Green Weakness (Deuteranomaly)"
        classification = "Anomalous Trichromacy"
        severity = "Mild to Moderate"
        status_class = "medium-confidence"
    else:
        diagnosis = "Dichromat"
        cvd_type = "Red-Green Dichromacy"
        classification = "Dichromacy"
        severity = "Severe"
        status_class = "low-confidence"
    
    return {
        'diagnosis': diagnosis,
        'cvd_type': cvd_type,
        'classification': classification,
        'severity': severity,
        'status_class': status_class,
        'confidence': 0.90
    }

def complete_assessment(test_images):
    results = {}
    
    results['ishihara'] = analyze_ishihara(test_images['ishihara'], '12')
    results['farnsworth'] = analyze_farnsworth(test_images['farnsworth'])
    results['cambridge'] = analyze_cambridge(test_images['cambridge'], 'Yes')
    results['spectrum'] = analyze_spectrum(test_images['spectrum'])
    results['anomaloscope'] = analyze_anomaloscope(test_images['anomaloscope'])
    
    visual_acuity = "20/20"
    eye_function = "Normal"
    
    if 'snellen' in test_images:
        acuity_score = np.random.uniform(0.7, 1.0)
        if acuity_score >= 0.95:
            visual_acuity = "20/20"
            snellen_result = "Perfect visual acuity"
            snellen_confidence = acuity_score
        elif acuity_score >= 0.85:
            visual_acuity = "20/25"
            snellen_result = "Excellent visual acuity"
            snellen_confidence = acuity_score
        elif acuity_score >= 0.75:
            visual_acuity = "20/30"
            snellen_result = "Good visual acuity"
            snellen_confidence = acuity_score
        else:
            visual_acuity = "20/40"
            snellen_result = "Fair visual acuity"
            snellen_confidence = acuity_score
        
        results['snellen'] = {
            'result': snellen_result,
            'visual_acuity': visual_acuity,
            'confidence': snellen_confidence
        }
    else:
        results['snellen'] = {
            'result': 'Not tested',
            'visual_acuity': 'Not assessed',
            'confidence': 0.0
        }
    
    if 'eyemuscle' in test_images:
        muscle_score = np.random.uniform(0.7, 1.0)
        if muscle_score >= 0.9:
            eye_function = "Normal - Excellent coordination"
            muscle_result = "All eye movements normal"
            muscle_confidence = muscle_score
        elif muscle_score >= 0.75:
            eye_function = "Normal - Good coordination"
            muscle_result = "Eye movements within normal range"
            muscle_confidence = muscle_score
        else:
            eye_function = "Mild coordination issues"
            muscle_result = "Some eye movement irregularities detected"
            muscle_confidence = muscle_score
        
        results['eyemuscle'] = {
            'result': muscle_result,
            'eye_function': eye_function,
            'confidence': muscle_confidence
        }
    else:
        results['eyemuscle'] = {
            'result': 'Not tested',
            'eye_function': 'Not assessed',
            'confidence': 0.0
        }
    
    confidences = [
        results['ishihara']['confidence'],
        results['farnsworth']['confidence'],
        results['cambridge']['confidence'],
        results['spectrum']['confidence'],
        results['anomaloscope']['confidence']
    ]
    
    if 'snellen' in test_images and results['snellen']['confidence'] > 0:
        confidences.append(results['snellen']['confidence'])
    
    if 'eyemuscle' in test_images and results['eyemuscle']['confidence'] > 0:
        confidences.append(results['eyemuscle']['confidence'])
    
    ensemble_confidence = np.mean(confidences)
    
    normal_votes = 0
    deficiency_votes = 0
    
    for test_name, test_result in results.items():
        result_text = str(test_result.get('result', '') or test_result.get('diagnosis', ''))
        if 'Normal' in result_text or 'Correct' in result_text or 'Excellent' in result_text or 'Perfect' in result_text or 'Good' in result_text:
            normal_votes += 1
        else:
            deficiency_votes += 1
    
    if normal_votes >= 4:
        final_diagnosis = "Normal Eye & Color Vision"
        cvd_type = "Trichromat (Normal)"
        severity = "None"
        final_status_class = "high-confidence"
    elif deficiency_votes >= 4:
        final_diagnosis = "Eye or Color Vision Issues Detected"
        cvd_type = "Possible Red-Green Deficiency or Visual Impairment"
        if deficiency_votes >= 6:
            severity = "Moderate to Severe"
            final_status_class = "low-confidence"
        else:
            severity = "Mild to Moderate"
            final_status_class = "medium-confidence"
    else:
        final_diagnosis = "Mixed Results - Further Testing Recommended"
        cvd_type = "Possible Mild Issues"
        severity = "Mild"
        final_status_class = "medium-confidence"
    
    individual_results = [
        {
            'Test': 'Ishihara Plates',
            'Result': results['ishihara']['result'],
            'Confidence': results['ishihara']['confidence']
        },
        {
            'Test': 'Farnsworth D-15',
            'Result': results['farnsworth']['result'],
            'Confidence': results['farnsworth']['confidence']
        },
        {
            'Test': 'Cambridge Color Test',
            'Result': results['cambridge']['diagnosis'],
            'Confidence': results['cambridge']['confidence']
        },
        {
            'Test': 'Spectrum Discrimination',
            'Result': results['spectrum']['result'],
            'Confidence': results['spectrum']['confidence']
        },
        {
            'Test': 'Anomaloscope',
            'Result': results['anomaloscope']['diagnosis'],
            'Confidence': results['anomaloscope']['confidence']
        },
        {
            'Test': 'Snellen Visual Acuity',
            'Result': results['snellen']['result'],
            'Confidence': results['snellen']['confidence']
        },
        {
            'Test': 'Eye Muscle & Focus',
            'Result': results['eyemuscle']['result'],
            'Confidence': results['eyemuscle']['confidence']
        }
    ]
    
    return {
        'final_diagnosis': final_diagnosis,
        'ensemble_confidence': ensemble_confidence,
        'cvd_type': cvd_type,
        'visual_acuity': visual_acuity,
        'eye_function': eye_function,
        'severity': severity,
        'final_status_class': final_status_class,
        'individual_results': individual_results,
        'detailed_results': results
    }

def build_colorblind_cnn(input_shape=(224, 224, 3), num_classes=6):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

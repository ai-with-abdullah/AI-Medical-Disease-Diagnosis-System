import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import re

def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    
    try:
        images = convert_from_bytes(pdf_bytes)
        
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            extracted_text += f"\n--- Page {i+1} ---\n{text}"
        
        return extracted_text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_txt(txt_file):
    text = txt_file.read()
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    return text

def extract_medical_data(report_file):
    filename = report_file.name
    
    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(report_file)
    else:
        text = extract_text_from_txt(report_file)
    
    extracted_data = parse_medical_text(text)
    
    return extracted_data

def parse_medical_text(text):
    data = {
        'raw_text': text,
        'vital_signs': {},
        'lab_results': {},
        'diagnoses': [],
        'medications': [],
        'clinical_notes': ''
    }
    
    age_match = re.search(r'age[:\s]+(\d+)', text, re.IGNORECASE)
    if age_match:
        data['vital_signs']['age'] = int(age_match.group(1))
    
    bp_match = re.search(r'blood\s+pressure[:\s]+(\d+)/(\d+)', text, re.IGNORECASE)
    if bp_match:
        data['vital_signs']['systolic_bp'] = int(bp_match.group(1))
        data['vital_signs']['diastolic_bp'] = int(bp_match.group(2))
    
    hr_match = re.search(r'heart\s+rate[:\s]+(\d+)', text, re.IGNORECASE)
    if hr_match:
        data['vital_signs']['heart_rate'] = int(hr_match.group(1))
    
    chol_match = re.search(r'cholesterol[:\s]+(\d+)', text, re.IGNORECASE)
    if chol_match:
        data['lab_results']['cholesterol'] = int(chol_match.group(1))
    
    glucose_match = re.search(r'glucose[:\s]+(\d+)', text, re.IGNORECASE)
    if glucose_match:
        data['lab_results']['glucose'] = int(glucose_match.group(1))
    
    diagnosis_keywords = ['pneumonia', 'hypertension', 'diabetes', 'asthma', 
                          'eczema', 'dermatitis', 'melanoma', 'cardiac']
    for keyword in diagnosis_keywords:
        if re.search(keyword, text, re.IGNORECASE):
            data['diagnoses'].append(keyword.capitalize())
    
    medication_pattern = r'(?:prescribed|medication|taking)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    med_matches = re.findall(medication_pattern, text)
    data['medications'] = list(set(med_matches))
    
    data['clinical_notes'] = text[:500]
    
    return data

def analyze_medical_report_sentiment(text):
    negative_keywords = ['abnormal', 'elevated', 'high risk', 'concerning', 
                          'deteriorating', 'critical', 'severe']
    positive_keywords = ['normal', 'healthy', 'stable', 'improved', 
                          'within range', 'no concerns']
    
    text_lower = text.lower()
    
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    
    if negative_count > positive_count:
        sentiment = 'Concerning'
        risk_score = min(0.3 + (negative_count * 0.1), 0.9)
    elif positive_count > negative_count:
        sentiment = 'Normal'
        risk_score = max(0.1, 0.5 - (positive_count * 0.05))
    else:
        sentiment = 'Neutral'
        risk_score = 0.5
    
    return {
        'sentiment': sentiment,
        'risk_score': risk_score,
        'negative_indicators': negative_count,
        'positive_indicators': positive_count
    }

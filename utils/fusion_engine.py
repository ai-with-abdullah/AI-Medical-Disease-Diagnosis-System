import numpy as np
from models.pneumonia_model import analyze_xray_image
from models.skin_model import analyze_skin_image
from models.heart_model import predict_heart_disease
from models.audio_model import analyze_audio
from utils.nlp_processor import extract_medical_data, analyze_medical_report_sentiment

def multimodal_fusion(inputs):
    disease_type = inputs['disease_type']
    fusion_method = inputs['fusion_method']
    
    modality_predictions = []
    modality_confidences = []
    modality_names = []
    
    if inputs['image']:
        if disease_type == "Pneumonia":
            image_result = analyze_xray_image(inputs['image'], 'Ensemble (All Models)')
            modality_predictions.append(image_result['prediction'])
            modality_confidences.append(image_result['confidence'])
            modality_names.append('X-Ray Image')
        elif disease_type == "Skin Disease":
            image_result = analyze_skin_image(inputs['image'], 'Ensemble')
            modality_predictions.append(image_result['disease'])
            modality_confidences.append(image_result['confidence'])
            modality_names.append('Skin Image')
    
    if inputs['audio']:
        audio_result = analyze_audio(inputs['audio'])
        modality_predictions.append(audio_result['prediction'])
        modality_confidences.append(audio_result['confidence'])
        modality_names.append('Audio Analysis')
    
    if inputs['report']:
        report_data = extract_medical_data(inputs['report'])
        report_sentiment = analyze_medical_report_sentiment(report_data['raw_text'])
        
        if report_sentiment['sentiment'] == 'Concerning':
            report_prediction = f"Possible {disease_type}"
        else:
            report_prediction = "Normal"
        
        modality_predictions.append(report_prediction)
        modality_confidences.append(1.0 - report_sentiment['risk_score'])
        modality_names.append('Medical Report')
    
    if fusion_method == "Weighted Average":
        final_confidence, final_diagnosis, fusion_weights = weighted_average_fusion(
            modality_predictions, modality_confidences
        )
    elif fusion_method == "Voting Ensemble":
        final_confidence, final_diagnosis, fusion_weights = voting_ensemble_fusion(
            modality_predictions, modality_confidences
        )
    elif fusion_method == "Bayesian Inference":
        final_confidence, final_diagnosis, fusion_weights = bayesian_fusion(
            modality_predictions, modality_confidences
        )
    else:
        final_confidence, final_diagnosis, fusion_weights = stacking_fusion(
            modality_predictions, modality_confidences
        )
    
    modality_results = [
        {
            'Modality': name,
            'Prediction': pred,
            'Confidence': f"{conf:.2%}"
        }
        for name, pred, conf in zip(modality_names, modality_predictions, modality_confidences)
    ]
    
    return {
        'diagnosis': final_diagnosis,
        'final_confidence': final_confidence,
        'fusion_method': fusion_method,
        'modalities_count': len(modality_names),
        'modality_results': modality_results,
        'fusion_weights': fusion_weights
    }

def weighted_average_fusion(predictions, confidences):
    total_confidence = sum(confidences)
    weights = [conf / total_confidence for conf in confidences]
    
    final_confidence = np.average(confidences, weights=confidences)
    
    prediction_scores = {}
    for pred, weight in zip(predictions, weights):
        if pred in prediction_scores:
            prediction_scores[pred] += weight
        else:
            prediction_scores[pred] = weight
    
    final_diagnosis = max(prediction_scores, key=prediction_scores.get)
    
    return final_confidence, final_diagnosis, weights

def voting_ensemble_fusion(predictions, confidences):
    from collections import Counter
    
    prediction_counts = Counter(predictions)
    final_diagnosis = prediction_counts.most_common(1)[0][0]
    
    final_confidence = np.mean(confidences)
    
    weights = [1.0 / len(predictions)] * len(predictions)
    
    return final_confidence, final_diagnosis, weights

def bayesian_fusion(predictions, confidences):
    prior = 0.5
    
    likelihood_product = 1.0
    for conf in confidences:
        likelihood_product *= conf
    
    posterior = (likelihood_product * prior) / (likelihood_product * prior + (1 - likelihood_product) * (1 - prior))
    
    final_confidence = posterior
    
    prediction_scores = {}
    for pred, conf in zip(predictions, confidences):
        if pred in prediction_scores:
            prediction_scores[pred] *= conf
        else:
            prediction_scores[pred] = conf
    
    final_diagnosis = max(prediction_scores, key=prediction_scores.get)
    
    weights = [conf / sum(confidences) for conf in confidences]
    
    return final_confidence, final_diagnosis, weights

def stacking_fusion(predictions, confidences):
    confidence_weights = np.array(confidences)
    confidence_weights = confidence_weights ** 2
    confidence_weights = confidence_weights / np.sum(confidence_weights)
    
    final_confidence = np.sum(np.array(confidences) * confidence_weights)
    
    prediction_scores = {}
    for pred, weight in zip(predictions, confidence_weights):
        if pred in prediction_scores:
            prediction_scores[pred] += weight
        else:
            prediction_scores[pred] = weight
    
    final_diagnosis = max(prediction_scores, key=prediction_scores.get)
    
    return final_confidence, final_diagnosis, confidence_weights.tolist()

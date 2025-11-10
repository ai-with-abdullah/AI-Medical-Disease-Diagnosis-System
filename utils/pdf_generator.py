from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
from datetime import datetime

def generate_diagnosis_report(result):
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    
    story.append(Paragraph("AI Multi-Modal Disease Detection", title_style))
    story.append(Paragraph("Comprehensive Diagnostic Report", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {current_time}", normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Diagnosis Summary", heading_style))
    
    diagnosis_data = [
        ['Final Diagnosis:', result['diagnosis']],
        ['Confidence Level:', f"{result['final_confidence']:.2%}"],
        ['Fusion Method:', result['fusion_method']],
        ['Modalities Analyzed:', str(result['modalities_count'])]
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2.5*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(diagnosis_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Individual Modality Results", heading_style))
    
    modality_data = [['Modality', 'Prediction', 'Confidence']]
    for item in result['modality_results']:
        modality_data.append([
            item['Modality'],
            item['Prediction'],
            item['Confidence']
        ])
    
    modality_table = Table(modality_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
    modality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(modality_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Clinical Recommendations", heading_style))
    
    if result['final_confidence'] > 0.8:
        recommendation_text = f"""
        Based on the high confidence level ({result['final_confidence']:.2%}) of this multi-modal analysis, 
        the diagnosis of <b>{result['diagnosis']}</b> is strongly indicated. We recommend:
        <br/><br/>
        • Consult with a healthcare professional for clinical confirmation<br/>
        • Follow up with appropriate medical tests as recommended<br/>
        • Keep this report for your medical records<br/>
        • Monitor symptoms and seek immediate care if condition worsens
        """
    elif result['final_confidence'] > 0.6:
        recommendation_text = f"""
        This analysis shows moderate confidence ({result['final_confidence']:.2%}) for <b>{result['diagnosis']}</b>. 
        We recommend:
        <br/><br/>
        • Schedule an appointment with a healthcare provider for further evaluation<br/>
        • Additional diagnostic tests may be necessary for confirmation<br/>
        • Continue monitoring symptoms<br/>
        • Maintain a healthy lifestyle and follow medical advice
        """
    else:
        recommendation_text = f"""
        This analysis has lower confidence ({result['final_confidence']:.2%}). The prediction of 
        <b>{result['diagnosis']}</b> should be interpreted with caution. We recommend:
        <br/><br/>
        • Seek professional medical consultation immediately<br/>
        • Additional comprehensive testing is strongly advised<br/>
        • Do not rely solely on this AI-based analysis<br/>
        • Consider second opinions from medical specialists
        """
    
    story.append(Paragraph(recommendation_text, normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    <b>Medical Disclaimer:</b> This report is generated by an AI-based diagnostic system for 
    educational and assistive purposes only. It should NOT be used as a substitute for professional 
    medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers 
    with any questions regarding medical conditions. Never disregard professional medical advice or 
    delay seeking it because of information from this report.
    <br/><br/>
    <b>System Information:</b> This analysis uses ensemble machine learning models including CNNs, 
    Random Forest classifiers, and multi-modal fusion algorithms. The system has been trained on 
    publicly available medical datasets but has not been clinically validated or FDA approved.
    """
    
    story.append(Paragraph(disclaimer_text, normal_style))
    story.append(Spacer(1, 20))
    
    footer_text = """
    <br/><br/>
    ---<br/>
    AI Multi-Modal Disease Detection System<br/>
    Powered by TensorFlow, Scikit-learn, and Advanced Fusion Algorithms<br/>
    For questions or concerns, please contact your healthcare provider.
    """
    story.append(Paragraph(footer_text, normal_style))
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer

def generate_colorblind_report(result):
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    
    story.append(Paragraph("Comprehensive Color Vision Assessment", title_style))
    story.append(Paragraph("5-Test Ensemble Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Assessment Date:</b> {current_time}", normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Final Diagnosis", heading_style))
    
    diagnosis_data = [
        ['Diagnosis:', result['final_diagnosis']],
        ['Color Vision Type:', result['cvd_type']],
        ['Severity Level:', result['severity']],
        ['Ensemble Confidence:', f"{result['ensemble_confidence']:.2%}"]
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2.5*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(diagnosis_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Individual Test Results", heading_style))
    
    test_data = [['Test Name', 'Result', 'Confidence']]
    for item in result['individual_results']:
        test_data.append([
            item['Test'],
            item['Result'],
            f"{item['Confidence']:.2%}"
        ])
    
    test_table = Table(test_data, colWidths=[2.2*inch, 2.8*inch, 1.5*inch])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(test_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Understanding Your Results", heading_style))
    
    if result['severity'] == "None":
        explanation = """
        Your comprehensive color vision assessment indicates <b>normal color vision (trichromacy)</b>. 
        All five tests showed consistent results indicating that your color perception is within 
        normal parameters. You have normal red, green, and blue cone function.
        """
    elif "Mild" in result['severity']:
        explanation = f"""
        Your assessment indicates <b>{result['cvd_type']}</b> with mild severity. This means you 
        have some difficulty distinguishing certain colors, but it's relatively minor. Many people 
        with mild color vision deficiencies lead normal lives with minimal impact.
        <br/><br/>
        <b>Recommendations:</b><br/>
        • Be aware of your color vision limitations in certain situations<br/>
        • Use color-blind friendly tools and apps when needed<br/>
        • Inform employers if color discrimination is critical for your work<br/>
        • Consider follow-up testing if symptoms worsen
        """
    else:
        explanation = f"""
        Your assessment indicates <b>{result['cvd_type']}</b> with {result['severity'].lower()} severity. 
        This suggests significant difficulty in color discrimination. Multiple tests confirmed this finding, 
        indicating a consistent pattern.
        <br/><br/>
        <b>Recommendations:</b><br/>
        • Consult with an ophthalmologist or optometrist for clinical confirmation<br/>
        • Learn about color-blind friendly tools and technologies<br/>
        • Consider special lenses or filters designed for color vision deficiency<br/>
        • Inform relevant parties (employers, educators) about your color vision status
        """
    
    story.append(Paragraph(explanation, normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Test Descriptions", heading_style))
    test_descriptions = """
    <b>1. Ishihara Plates Test:</b> Classic color blindness test using numbered patterns to detect 
    red-green deficiencies.<br/><br/>
    <b>2. Farnsworth D-15 Test:</b> Color arrangement test that evaluates color discrimination ability 
    across the spectrum.<br/><br/>
    <b>3. Cambridge Color Test:</b> Measures color discrimination thresholds using pattern detection 
    in varying chromatic contrasts.<br/><br/>
    <b>4. Color Spectrum Discrimination:</b> Tests ability to distinguish subtle color variations 
    across different wavelengths.<br/><br/>
    <b>5. Anomaloscope Simulation:</b> Digital version of the gold-standard clinical test for 
    diagnosing color vision deficiencies.
    """
    story.append(Paragraph(test_descriptions, normal_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Important Information", heading_style))
    disclaimer = """
    <b>Disclaimer:</b> This automated color vision assessment is designed for screening purposes only 
    and should not replace professional eye examination. For definitive diagnosis and clinical 
    management, please consult with a qualified eye care professional (optometrist or ophthalmologist).
    <br/><br/>
    <b>Test Accuracy:</b> Results may be affected by screen calibration, lighting conditions, and 
    individual variations. Clinical testing under controlled conditions is recommended for accurate 
    diagnosis.
    """
    story.append(Paragraph(disclaimer, normal_style))
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json
import datetime
from io import BytesIO
import base64

# Try to import OpenCV with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available. Using PIL-based color analysis.")
    cv2 = None
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. PDF generation will be disabled.")
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="ER+ Breast Cancer Risk Monitor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language translations
LANGUAGES = {
    "English": {
        "title": "ER+ Breast Cancer Risk Monitoring & Support AI",
        "subtitle": "AI-Powered Multi-Factor Risk Assessment for Estrogen Receptor Positive Breast Cancer",
        "home": "Home",
        "analyzer": "Multi-Image Analyzer",
        "tracker": "Progress Tracker",
        "family": "Family History Assessment",
        "symptoms": "AI Symptom Analysis",
        "chat": "Support Chat",
        "resources": "Health Resources",
        "trials": "Clinical Trials",
        "education": "ER+ Education",
        "export": "Data Export",
        "welcome": "Welcome to ER+ Breast Cancer Risk Monitoring",
        "upload_image": "Upload LFA Test Strip Image",
        "analyze": "Analyze Image",
        "risk_low": "Low Risk",
        "risk_moderate": "Moderate Risk",
        "risk_high": "High Risk",
        "calibrate": "Calibrate Colors",
        "reminder": "Test Reminder",
        "clinic_finder": "Find Nearby Clinics"
    },
    "Filipino": {
        "title": "ER+ Breast Cancer Risk Monitor",
        "subtitle": "AI para sa Pagsubaybay ng Panganib sa ER+ Breast Cancer",
        "home": "Tahanan",
        "analyzer": "Multi-Image Analyzer",
        "tracker": "Progress Tracker",
        "family": "Family History Assessment",
        "symptoms": "AI Symptom Analysis",
        "chat": "Support Chat",
        "resources": "Health Resources",
        "trials": "Clinical Trials",
        "education": "ER+ Edukasyon",
        "export": "Data Export",
        "welcome": "Maligayang pagdating sa ER+ Breast Cancer Risk Monitor",
        "upload_image": "Mag-upload ng LFA Test Strip Image",
        "analyze": "Suriin ang Larawan",
        "risk_low": "Mababang Panganib",
        "risk_moderate": "Katamtamang Panganib",
        "risk_high": "Mataas na Panganib",
        "calibrate": "I-calibrate ang Kulay",
        "reminder": "Test Reminder",
        "clinic_finder": "Maghanap ng Malapit na Clinic"
    },
    "Spanish": {
        "title": "Monitor de Riesgo de Cáncer de Mama ER+",
        "subtitle": "Evaluación de Riesgo Multi-Factor con IA para Cáncer de Mama ER+",
        "home": "Inicio",
        "analyzer": "Analizador Multi-Imagen",
        "tracker": "Seguimiento de Progreso",
        "family": "Evaluación de Historia Familiar",
        "symptoms": "Análisis de Síntomas con IA",
        "chat": "Chat de Apoyo",
        "resources": "Recursos de Salud",
        "trials": "Ensayos Clínicos",
        "education": "Educación ER+",
        "export": "Exportar Datos",
        "welcome": "Bienvenido al Monitor de Riesgo de Cáncer de Mama ER+",
        "upload_image": "Subir Imagen de Tira de Prueba",
        "analyze": "Analizar Imagen",
        "risk_low": "Riesgo Bajo",
        "risk_moderate": "Riesgo Moderado",
        "risk_high": "Riesgo Alto",
        "calibrate": "Calibrar Colores",
        "reminder": "Recordatorio de Prueba",
        "clinic_finder": "Encontrar Clínicas Cercanas"
    }
}

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'family_history' not in st.session_state:
    st.session_state.family_history = {}
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'biomarker_history' not in st.session_state:
    st.session_state.biomarker_history = []
if 'last_test_date' not in st.session_state:
    st.session_state.last_test_date = None
if 'calibration_reference' not in st.session_state:
    st.session_state.calibration_reference = None
if 'user_location' not in st.session_state:
    st.session_state.user_location = {"city": "", "barangay": ""}

def get_text(key):
    return LANGUAGES[st.session_state.language].get(key, key)

def analyze_er_image_with_confidence(image, calibration_ref=None):
    """Enhanced ER analysis with confidence levels for ER+ cancer detection"""
    img_array = np.array(image)
    
    # Color calibration if reference is provided
    if calibration_ref is not None:
        calibration_factor = calculate_calibration_factor(img_array, calibration_ref)
    else:
        calibration_factor = 1.0
    
    if CV2_AVAILABLE and cv2 is not None:
        # Use OpenCV for color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Analyze red spectrum for ER detection
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Calculate red intensity
        total_pixels = img_array.shape[0] * img_array.shape[1]
        red_pixels = np.sum(red_mask > 0)
        red_intensity = (red_pixels / total_pixels) * calibration_factor
        
        # Calculate average red values for confidence
        red_areas = img_array[red_mask > 0]
        if len(red_areas) > 0:
            avg_red_value = np.mean(red_areas[:, 0])  # R channel
            color_saturation = np.mean(hsv[red_mask > 0, 1])  # S channel
        else:
            avg_red_value = 0
            color_saturation = 0
    else:
        # Fallback PIL-based color analysis when OpenCV is not available
        # Convert RGB to HSV manually
        def rgb_to_hsv(rgb):
            r, g, b = rgb / 255.0
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)
            diff = max_val - min_val
            
            # Hue calculation
            h = np.zeros_like(max_val)
            mask = diff != 0
            
            # Red is max
            red_mask = (max_val == r) & mask
            h[red_mask] = (60 * ((g[red_mask] - b[red_mask]) / diff[red_mask]) + 360) % 360
            
            # Green is max
            green_mask = (max_val == g) & mask
            h[green_mask] = (60 * ((b[green_mask] - r[green_mask]) / diff[green_mask]) + 120) % 360
            
            # Blue is max
            blue_mask = (max_val == b) & mask
            h[blue_mask] = (60 * ((r[blue_mask] - g[blue_mask]) / diff[blue_mask]) + 240) % 360
            
            # Saturation
            s = np.zeros_like(max_val)
            s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]
            
            # Value
            v = max_val
            
            return np.stack([h, s * 255, v * 255], axis=-1)
        
        hsv = rgb_to_hsv(img_array.astype(np.float32))
        
        # Create red mask using HSV thresholds
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        # Red hue ranges (0-10 and 350-360 degrees, converted to 0-180 scale)
        red_mask1 = ((h >= 0) & (h <= 10)) & (s >= 50) & (v >= 50)
        red_mask2 = ((h >= 170) & (h <= 180)) & (s >= 50) & (v >= 50)
        red_mask = red_mask1 | red_mask2
        
        # Also check for high red values in RGB
        r_channel = img_array[:, :, 0].astype(np.float32)
        g_channel = img_array[:, :, 1].astype(np.float32)
        b_channel = img_array[:, :, 2].astype(np.float32)
        
        # Red dominance mask (red significantly higher than green and blue)
        red_dominance = (r_channel > (g_channel + 30)) & (r_channel > (b_channel + 30)) & (r_channel > 100)
        
        # Combine masks
        final_red_mask = red_mask | red_dominance
        
        # Calculate red intensity
        total_pixels = img_array.shape[0] * img_array.shape[1]
        red_pixels = np.sum(final_red_mask)
        red_intensity = (red_pixels / total_pixels) * calibration_factor
        
        # Calculate average red values for confidence
        if red_pixels > 0:
            avg_red_value = np.mean(r_channel[final_red_mask])
            # Use intensity difference as saturation proxy
            red_areas = img_array[final_red_mask]
            color_saturation = np.mean(np.max(red_areas, axis=1) - np.min(red_areas, axis=1)) if len(red_areas) > 0 else 0
        else:
            avg_red_value = np.mean(r_channel)  # Use overall red average
            color_saturation = 50  # Default moderate saturation
    
    # Confidence calculation based on color intensity and saturation
    confidence_factors = []
    
    # Factor 1: Red intensity coverage
    if red_intensity > 0.15:
        confidence_factors.append(0.95)
    elif red_intensity > 0.08:
        confidence_factors.append(0.80)
    elif red_intensity > 0.02:
        confidence_factors.append(0.65)
    else:
        confidence_factors.append(0.50)
    
    # Factor 2: Color saturation
    if color_saturation > 150:
        confidence_factors.append(0.90)
    elif color_saturation > 100:
        confidence_factors.append(0.75)
    elif color_saturation > 50:
        confidence_factors.append(0.60)
    else:
        confidence_factors.append(0.40)
    
    # Factor 3: Red value intensity
    if avg_red_value > 200:
        confidence_factors.append(0.95)
    elif avg_red_value > 150:
        confidence_factors.append(0.80)
    elif avg_red_value > 100:
        confidence_factors.append(0.65)
    else:
        confidence_factors.append(0.45)
    
    # Calculate overall confidence
    confidence = np.mean(confidence_factors)
    
    # Determine ER status and risk level based on user specifications
    if red_intensity < 0.02:
        er_status = "ER Negative"
        risk_level = "Low Risk (0-10%)"
        risk_score = red_intensity * 500  # 0-10%
        color_description = "No color detected"
    elif red_intensity < 0.08:
        er_status = "ER Low Positive"
        risk_level = "Moderate Risk"
        risk_score = 30 + (red_intensity - 0.02) * 333  # 30-50%
        color_description = "Faint red coloration"
    else:
        er_status = "ER High Positive"
        risk_level = "High Risk"
        risk_score = 60 + (red_intensity - 0.08) * 300  # 60-90%
        color_description = "Dark red coloration"
    
    # Cap risk score at 90%
    risk_score = min(risk_score, 90)
    
    return {
        'er_intensity': red_intensity * 100,
        'er_status': er_status,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'confidence': confidence * 100,
        'color_description': color_description,
        'avg_red_value': avg_red_value,
        'color_saturation': color_saturation
    }

def calculate_calibration_factor(image, reference_color):
    """Calculate calibration factor based on reference color"""
    # Simplified calibration - in practice, would use color science
    expected_red = [255, 0, 0]  # Expected red reference
    actual_red = np.mean(reference_color, axis=(0, 1))
    
    # Calculate calibration factor
    factor = np.mean(expected_red) / np.mean(actual_red) if np.mean(actual_red) > 0 else 1.0
    return np.clip(factor, 0.5, 2.0)  # Limit calibration range

def multi_factor_risk_fusion(er_results, symptoms_data, family_data, test_frequency):
    """Advanced multi-factor risk fusion algorithm focused on ER"""
    
    # ER risk (50% weight) - increased weight since it's the primary focus
    er_risk = er_results['risk_score'] / 100
    
    # Symptom risk (25% weight)
    symptom_risk = calculate_symptom_risk_score(symptoms_data)
    
    # Family history risk (20% weight)
    family_risk = calculate_family_risk_modifier(family_data)
    
    # Test frequency bonus (5% weight)
    frequency_bonus = calculate_frequency_bonus(test_frequency)
    
    # Fusion calculation
    composite_risk = (
        er_risk * 0.5 +
        symptom_risk * 0.25 +
        family_risk * 0.2 +
        frequency_bonus * 0.05
    )
    
    # Determine risk level
    if composite_risk < 0.3:
        return "Low Risk", composite_risk * 100, "green"
    elif composite_risk < 0.6:
        return "Moderate Risk", composite_risk * 100, "orange"
    else:
        return "High Risk", composite_risk * 100, "red"

def calculate_symptom_risk_score(symptoms_data):
    """Calculate normalized symptom risk score"""
    if not symptoms_data:
        return 0.1
    
    risk_weights = {
        "lumps": 0.3,
        "skin_changes": 0.2,
        "nipple_discharge": 0.25,
        "pain": 0.1,
        "size_changes": 0.15
    }
    
    total_score = 0
    for symptom, severity in symptoms_data.items():
        if symptom in risk_weights:
            total_score += risk_weights[symptom] * (severity / 5.0)
    
    return min(total_score, 1.0)

def calculate_family_risk_modifier(family_data):
    """Calculate family history risk modifier"""
    if not family_data:
        return 0.1
    
    risk_score = 0.1
    
    if family_data.get('mother_cancer', False):
        risk_score += 0.3
    if family_data.get('sister_cancer', False):
        risk_score += 0.2
    if family_data.get('brca_positive', False):
        risk_score += 0.4
    if family_data.get('early_onset', False):
        risk_score += 0.2
    
    return min(risk_score, 1.0)

def calculate_frequency_bonus(test_frequency):
    """Calculate bonus for regular testing"""
    if test_frequency >= 4:  # 4+ times per year
        return 0.1
    elif test_frequency >= 2:  # 2-3 times per year
        return 0.05
    else:
        return 0

def check_test_reminder():
    """Check if user needs a test reminder"""
    if st.session_state.last_test_date:
        last_test = datetime.datetime.strptime(st.session_state.last_test_date, "%Y-%m-%d")
        days_since = (datetime.datetime.now() - last_test).days
        
        if days_since >= 90:  # 3 months
            return True, days_since
    return False, 0

def get_adherence_score():
    """Calculate adherence score based on testing frequency"""
    if len(st.session_state.risk_history) < 2:
        return 0
    
    # Calculate average days between tests
    dates = [datetime.datetime.strptime(r['date'], "%Y-%m-%d %H:%M") for r in st.session_state.risk_history]
    dates.sort()
    
    if len(dates) < 2:
        return 0
    
    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    avg_interval = sum(intervals) / len(intervals)
    
    # Score based on ideal 90-day interval
    if avg_interval <= 90:
        return 100
    elif avg_interval <= 120:
        return 80
    elif avg_interval <= 180:
        return 60
    else:
        return 40

def generate_pdf_report():
    """Generate PDF health report"""
    if not REPORTLAB_AVAILABLE:
        # Create a simple text report instead
        report_text = f"""
ER+ BREAST CANCER RISK ASSESSMENT REPORT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Total Assessments: {len(st.session_state.risk_history)}

Latest Results:
"""
        if st.session_state.risk_history:
            latest = st.session_state.risk_history[-1]
            report_text += f"Latest Risk Level: {latest['risk']}\n"
            report_text += f"Latest Score: {latest['score']}\n"
        
        adherence = get_adherence_score()
        report_text += f"Test Adherence Score: {adherence}%\n"
        
        if st.session_state.family_history:
            report_text += "\nFamily History Risk Factors:\n"
            for factor, value in st.session_state.family_history.items():
                if value:
                    report_text += f"• {factor.replace('_', ' ').title()}\n"
        
        report_text += "\nThis report is for educational purposes only and does not replace professional medical advice."
        
        buffer = BytesIO()
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer
    
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "ER+ Breast Cancer Risk Assessment Report")
    
    # User info
    p.setFont("Helvetica", 12)
    p.drawString(100, 720, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p.drawString(100, 700, f"Total Assessments: {len(st.session_state.risk_history)}")
    
    # Latest results
    if st.session_state.risk_history:
        latest = st.session_state.risk_history[-1]
        p.drawString(100, 680, f"Latest Risk Level: {latest['risk']}")
        p.drawString(100, 660, f"Latest Score: {latest['score']}")
    
    # Adherence score
    adherence = get_adherence_score()
    p.drawString(100, 640, f"Test Adherence Score: {adherence}%")
    
    # Family history
    if st.session_state.family_history:
        p.drawString(100, 620, "Family History Risk Factors:")
        y_pos = 600
        for factor, value in st.session_state.family_history.items():
            if value:
                p.drawString(120, y_pos, f"• {factor.replace('_', ' ').title()}")
                y_pos -= 20
    
    # Disclaimer
    p.setFont("Helvetica-Oblique", 10)
    p.drawString(100, 100, "This report is for educational purposes only and does not replace professional medical advice.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def get_nearby_clinics(city, barangay):
    """Get nearby clinics based on location"""
    clinics_db = {
        "Manila": {
            "Ermita": [
                {"name": "Manila Health Center", "phone": "(02) 8527-4567", "services": "Free mammogram, consultation", "cost": "FREE"},
                {"name": "Barangay Ermita Health Station", "phone": "(02) 8527-1234", "services": "Basic screening, referral", "cost": "FREE"}
            ],
            "Malate": [
                {"name": "Malate Health Center", "phone": "(02) 8525-7890", "services": "Women's health, cancer screening", "cost": "FREE"},
                {"name": "DOH Manila Clinic", "phone": "(02) 8525-3456", "services": "Comprehensive care, BRCA testing", "cost": "FREE"}
            ]
        },
        "Quezon City": {
            "Diliman": [
                {"name": "UP-PGH Diliman Extension", "phone": "(02) 8981-8500", "services": "Specialist referral, genetic counseling", "cost": "FREE"},
                {"name": "Barangay Diliman Health Center", "phone": "(02) 8929-1234", "services": "Basic screening, health education", "cost": "FREE"}
            ],
            "Cubao": [
                {"name": "Cubao Health Center", "phone": "(02) 8912-3456", "services": "Women's health, mammogram referral", "cost": "FREE"},
                {"name": "Gateway Medical Clinic", "phone": "(02) 8912-7890", "services": "Private screening, consultation", "cost": "LOW COST"}
            ]
        },
        "Cebu": {
            "Lahug": [
                {"name": "Cebu City Health Center - Lahug", "phone": "(032) 238-1234", "services": "Free screening, counseling", "cost": "FREE"},
                {"name": "Chong Hua Hospital", "phone": "(032) 255-8000", "services": "Comprehensive cancer care", "cost": "PRIVATE"}
            ]
        }
    }
    
    return clinics_db.get(city, {}).get(barangay, [])

def get_comprehensive_recommendations(risk_level, confidence, er_results):
    """Get comprehensive recommendations based on risk level and confidence"""
    
    recommendations = {
        "immediate_actions": [],
        "medical_facilities": [],
        "support_resources": [],
        "financial_assistance": [],
        "lifestyle_changes": [],
        "follow_up": []
    }
    
    if risk_level == "High Risk":
        recommendations["immediate_actions"] = [
            "🚨 URGENT: See a doctor within 1-2 weeks",
            "📞 Call your doctor today to schedule an appointment",
            "🏥 Go to emergency room if experiencing severe symptoms",
            "📝 Document all symptoms and test results",
            "🚫 Do not delay seeking medical attention"
        ]
        
        recommendations["medical_facilities"] = [
            {"name": "Philippine General Hospital", "phone": "(02) 8554-8400", "cost": "FREE/CHARITY", "services": "Comprehensive cancer care"},
            {"name": "National Kidney Institute", "phone": "(02) 8981-0300", "cost": "FREE", "services": "Cancer screening, treatment"},
            {"name": "Jose Reyes Memorial Hospital", "phone": "(02) 8711-9491", "cost": "FREE", "services": "Government hospital, cancer unit"},
            {"name": "East Avenue Medical Center", "phone": "(02) 8928-0611", "cost": "FREE/CHARITY", "services": "Cancer screening, referral"}
        ]
        
        recommendations["support_resources"] = [
            {"name": "Philippine Cancer Society", "phone": "(02) 8927-2394", "services": "Free counseling, support groups"},
            {"name": "Breast Cancer Support Philippines", "phone": "(02) 8426-7394", "services": "Peer support, patient navigation"},
            {"name": "Cancer Warriors Philippines", "phone": "(02) 8555-2267", "services": "Financial assistance, support groups"},
            {"name": "Hope for Tomorrow Foundation", "phone": "(02) 8734-5566", "services": "Treatment assistance, counseling"}
        ]
        
        recommendations["financial_assistance"] = [
            "💰 PhilHealth: Covers breast cancer treatment packages",
            "🏥 PCSO Medical Assistance: Individual medical assistance",
            "🎗️ Malasakit Centers: One-stop shop for medical assistance",
            "💝 Private foundations: ICanServe, Pink Ribbon Philippines",
            "🏛️ Local government assistance programs",
            "📋 Apply for 4Ps health benefits if eligible"
        ]
        
    elif risk_level == "Moderate Risk":
        recommendations["immediate_actions"] = [
            "⚠️ Schedule doctor consultation within 2-4 weeks",
            "📋 Request mammogram or ultrasound",
            "📝 Monitor symptoms closely",
            "🔄 Continue monthly self-examinations",
            "📞 Call if symptoms worsen"
        ]
        
        recommendations["medical_facilities"] = [
            {"name": "Nearest Barangay Health Station", "phone": "Ask barangay captain", "cost": "FREE", "services": "Basic screening, referral"},
            {"name": "City Health Office", "phone": "Contact city hall", "cost": "FREE", "services": "Women's health, screening"},
            {"name": "RHU (Rural Health Unit)", "phone": "Contact municipality", "cost": "FREE", "services": "Basic healthcare, referral"},
            {"name": "District Hospital", "phone": "Contact DOH", "cost": "FREE/LOW COST", "services": "Secondary care"}
        ]
        
        recommendations["support_resources"] = [
            {"name": "Barangay Health Workers", "phone": "Contact barangay", "services": "Health education, referral"},
            {"name": "Women's Health Support Groups", "phone": "(02) 8555-HELP", "services": "Peer support, education"},
            {"name": "Philippine Cancer Society", "phone": "(02) 8927-2394", "services": "Information, support"},
            {"name": "DOH Health Hotline", "phone": "1555", "services": "24/7 health information"}
        ]
        
    else:  # Low Risk
        recommendations["immediate_actions"] = [
            "✅ Continue regular health monitoring",
            "📅 Schedule routine check-up in 6 months",
            "🔄 Continue monthly self-examinations",
            "📚 Learn about breast health",
            "🌱 Maintain healthy lifestyle"
        ]
        
        recommendations["medical_facilities"] = [
            {"name": "Barangay Health Station", "phone": "Contact barangay", "cost": "FREE", "services": "Routine check-ups, health education"},
            {"name": "Community Health Center", "phone": "Contact municipality", "cost": "FREE", "services": "Preventive care, education"},
            {"name": "Women's Health Clinic", "phone": "Contact city health", "cost": "FREE", "services": "Women's health services"}
        ]
        
        recommendations["support_resources"] = [
            {"name": "Healthy Lifestyle Support Groups", "phone": "Contact community center", "services": "Wellness programs"},
            {"name": "Women's Organizations", "phone": "Contact local NGOs", "services": "Health education, support"},
            {"name": "Online Health Communities", "phone": "N/A", "services": "Information, peer support"}
        ]
    
    # Common recommendations for all risk levels
    recommendations["lifestyle_changes"] = [
        "🥗 Eat a balanced diet rich in fruits and vegetables",
        "🏃‍♀️ Exercise regularly (150 minutes per week)",
        "🚭 Avoid smoking and limit alcohol",
        "⚖️ Maintain healthy weight",
        "😴 Get adequate sleep (7-8 hours)",
        "🧘‍♀️ Manage stress through relaxation techniques"
    ]
    
    recommendations["follow_up"] = [
        f"📅 Repeat testing in {'1-3 months' if risk_level == 'High Risk' else '3-6 months'}",
        "📱 Use this app to track symptoms and results",
        "👨‍⚕️ Share results with your healthcare provider",
        "📚 Stay informed about breast health",
        "🤝 Connect with support groups if needed"
    ]
    
    # Add government hotlines
    recommendations["emergency_hotlines"] = [
        {"name": "DOH Hotline", "phone": "1555", "available": "24/7"},
        {"name": "Emergency Services", "phone": "911", "available": "24/7"},
        {"name": "Philippine Cancer Society", "phone": "(02) 8927-2394", "available": "Business hours"},
        {"name": "Crisis Hotline", "phone": "(02) 8893-7603", "available": "24/7"}
    ]
    
    return recommendations

# Sidebar
st.sidebar.title("🎗️ ER+ Risk Monitor")

# Language selector
selected_language = st.sidebar.selectbox(
    "🌐 Language / Wika / Idioma",
    options=list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)
st.session_state.language = selected_language

# Test reminder check
needs_reminder, days_since = check_test_reminder()
if needs_reminder:
    st.sidebar.error(f"⏰ Test Reminder: {days_since} days since last test!")

# Navigation menu
page = st.sidebar.radio(
    "Menu",
    [
        get_text("home"),
        "🔬 Single Image Analyzer",
        get_text("analyzer"),
        get_text("tracker"),
        get_text("family"),
        get_text("symptoms"),
        get_text("chat"),
        get_text("resources"),
        get_text("trials"),
        get_text("education"),
        get_text("export")
    ]
)

# Adherence score display
if st.session_state.risk_history:
    adherence = get_adherence_score()
    st.sidebar.metric("Test Adherence", f"{adherence}%")

# Main content
st.title(get_text("title"))
st.caption(get_text("subtitle"))

# Enhanced disclaimer
st.error("⚠️ **Medical Disclaimer**: This application simulates ER+ breast cancer risk assessment for educational purposes only. Results are not clinically validated. Always consult healthcare professionals for medical advice, diagnosis, and treatment.")

if page == get_text("home"):
    st.header(get_text("welcome"))
    
    # Quick stats dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", len(st.session_state.risk_history))
    
    with col2:
        if st.session_state.risk_history:
            latest_risk = st.session_state.risk_history[-1]
            st.metric("Latest Risk", latest_risk['risk'])
    
    with col3:
        adherence = get_adherence_score()
        st.metric("Adherence Score", f"{adherence}%")
    
    with col4:
        family_factors = sum(1 for v in st.session_state.family_history.values() if v) if st.session_state.family_history else 0
        st.metric("Family Risk Factors", family_factors)
    
    # Feature overview
    st.subheader("🎯 ER+ Specific Features")
    
    feature_tabs = st.tabs(["🔬 Multi-Factor Analysis", "📊 Biomarker Tracking", "📅 Smart Reminders", "🏥 Clinic Finder"])
    
    with feature_tabs[0]:
        st.write("**Advanced Risk Fusion Algorithm**")
        st.write("- Combines image analysis, symptoms, family history, and test frequency")
        st.write("- Specialized for ER+ breast cancer risk factors")
        st.write("- Provides composite risk scoring")
        
        if st.button("🚀 Start Multi-Factor Analysis"):
            st.info("Click on 'Multi-Image Analyzer' in the sidebar to start the analysis.")
    
    with feature_tabs[1]:
        st.write("**Biomarker-Specific Tracking**")
        st.write("- Tracks ER, PR, and HER2 levels over time")
        st.write("- Dynamic threshold alerts")
        st.write("- Personalized trend analysis")
        
        if st.session_state.biomarker_history:
            # Mini biomarker chart
            df = pd.DataFrame(st.session_state.biomarker_history)
            fig = px.line(df, x='date', y=['ER', 'PR', 'HER2'], title="Biomarker Trends")
            st.plotly_chart(fig, use_container_width=True)
    
    with feature_tabs[2]:
        st.write("**Smart Test Reminders**")
        st.write("- Automated 3-month test reminders")
        st.write("- Adherence score tracking")
        st.write("- Personalized scheduling")
        
        if needs_reminder:
            st.warning(f"⏰ Next test due! {days_since} days since last assessment.")
        else:
            st.success("✅ You're up to date with testing!")
    
    with feature_tabs[3]:
        st.write("**Barangay Clinic Finder**")
        st.write("- Find nearby health centers")
        st.write("- Services and contact information")
        st.write("- Referral recommendations")
        
        user_city = st.selectbox("Select City", ["Manila", "Quezon City", "Cebu", "Other"])
        if user_city != "Other":
            st.session_state.user_location["city"] = user_city

elif page == "🔬 Single Image Analyzer":
    st.header("🔬 Single ER Image Analyzer")
    st.write("*Analyze a single test strip image for ER (Estrogen Receptor) status exclusively*")
    
    # Color calibration section
    with st.expander("🎨 Color Calibration (Optional)"):
        st.write("Upload a reference image with a known red color patch for better accuracy")
        calibration_file = st.file_uploader("Upload Color Reference", type=['png', 'jpg', 'jpeg'], key="single_cal")
        
        if calibration_file:
            cal_image = Image.open(calibration_file)
            st.image(cal_image, caption="Calibration Reference", width=200)
            st.session_state.calibration_reference = cal_image
            st.success("✅ Calibration reference set!")
    
    # Main image upload
    uploaded_file = st.file_uploader(
        "Upload ER Test Strip Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image showing ER test zone",
        key="single_upload"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.subheader("📸 Uploaded Image")
        st.image(image, caption="ER Test Strip", use_container_width=True)
        
        # Analysis button
        if st.button("🔬 Analyze ER Status", type="primary", key="single_analyze"):
            with st.spinner("Analyzing ER status..."):
                # Enhanced ER analysis
                er_results = analyze_er_image_with_confidence(
                    image, 
                    st.session_state.calibration_reference
                )
                
                # Debug information
                st.write(f"**Debug Info**: OpenCV Available: {CV2_AVAILABLE}")
                st.write(f"**Image Shape**: {np.array(image).shape}")
                st.write(f"**Average RGB Values**: R:{np.mean(np.array(image)[:,:,0]):.1f}, G:{np.mean(np.array(image)[:,:,1]):.1f}, B:{np.mean(np.array(image)[:,:,2]):.1f}")
                
                # Results section with organized layout
                st.markdown("---")
                st.subheader("🎯 ER Analysis Results")
                
                # Main results in organized columns
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    risk_color = "red" if "High Risk" in er_results['risk_level'] else "orange" if "Moderate Risk" in er_results['risk_level'] else "green"
                    st.markdown(f"### Risk Level")
                    st.markdown(f":{risk_color}[**{er_results['risk_level']}**]")
                
                with result_col2:
                    st.markdown(f"### Risk Score")
                    st.markdown(f"**{er_results['risk_score']:.1f}%**")
                
                with result_col3:
                    confidence_color = "green" if er_results['confidence'] > 80 else "orange" if er_results['confidence'] > 60 else "red"
                    st.markdown(f"### Confidence")
                    st.markdown(f":{confidence_color}[**{er_results['confidence']:.1f}%**]")
                
                # Detailed results
                st.markdown("---")
                st.subheader("📊 Detailed Analysis")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.metric("ER Status", er_results['er_status'])
                    st.metric("Color Intensity", f"{er_results['er_intensity']:.1f}%")
                    st.write(f"**Color Description**: {er_results['color_description']}")
                
                with detail_col2:
                    st.metric("Average Red Value", f"{er_results['avg_red_value']:.0f}")
                    st.metric("Color Saturation", f"{er_results['color_saturation']:.0f}")
                
                # Comprehensive Recommendations
                st.markdown("---")
                st.subheader("📋 Recommendations & Next Steps")
                
                recommendations = get_comprehensive_recommendations(
                    er_results['risk_level'].replace(" (0-10%)", ""), 
                    er_results['confidence'], 
                    er_results
                )
                
                # Immediate Actions
                with st.expander("🚨 Immediate Actions Required", expanded=True):
                    for action in recommendations["immediate_actions"]:
                        st.write(f"• {action}")
                
                # Medical Facilities
                with st.expander("🏥 Free & Low-Cost Medical Facilities"):
                    for facility in recommendations["medical_facilities"]:
                        st.write(f"**{facility['name']}**")
                        st.write(f"📞 {facility['phone']} | 💰 {facility['cost']}")
                        st.write(f"🩺 {facility['services']}")
                        st.write("---")
                
                # Save results
                result_entry = {
                    'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'risk': er_results['risk_level'],
                    'score': f"{er_results['risk_score']:.1f}%",
                    'confidence': f"{er_results['confidence']:.1f}%",
                    'type': 'Single ER Analysis',
                    'er_results': er_results
                }
                st.session_state.risk_history.append(result_entry)
                st.session_state.last_test_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # Save ER history
                er_entry = {
                    'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                    'ER': er_results['er_intensity'],
                    'PR': 0,  # Default value for PR since this is ER-only analysis
                    'HER2': 0,  # Default value for HER2 since this is ER-only analysis
                    'risk_level': er_results['risk_level'],
                    'confidence': er_results['confidence']
                }
                st.session_state.biomarker_history.append(er_entry)
                
                st.success("✅ Single ER Analysis complete! Results saved.")

elif page == get_text("analyzer"):
    st.header("📸 Multi-Image ER Analyzer")
    st.write("*Analyze multiple test strip images for ER (Estrogen Receptor) status comparison*")
    
    # Multiple image upload
    st.subheader("📸 Upload Multiple ER Test Images")
    uploaded_files = st.file_uploader(
        "Upload multiple ER test strip images for comparison",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload 2-5 images for comparison analysis"
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images")
        
        if len(uploaded_files) > 5:
            st.warning("⚠️ Please upload maximum 5 images for better analysis")
            uploaded_files = uploaded_files[:5]
        
        # Display uploaded images
        cols = st.columns(min(len(uploaded_files), 3))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {idx+1}", use_container_width=True)
        
        # Analysis button
        if st.button("🔬 Analyze All ER Images", type="primary"):
            with st.spinner("Analyzing all ER images..."):
                results = []
                
                # Analyze each image
                for idx, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    er_results = analyze_er_image_with_confidence(image, st.session_state.calibration_reference)
                    er_results['image_name'] = f"Image {idx+1}"
                    results.append(er_results)
                
                # Results comparison
                st.markdown("---")
                st.subheader("🎯 Multi-Image ER Analysis Results")
                
                # Summary table
                summary_data = []
                for result in results:
                    summary_data.append({
                        'Image': result['image_name'],
                        'Risk Level': result['risk_level'],
                        'Risk Score': f"{result['risk_score']:.1f}%",
                        'Confidence': f"{result['confidence']:.1f}%",
                        'ER Status': result['er_status'],
                        'Color Description': result['color_description']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Visual comparison
                st.subheader("📊 Risk Score Comparison")
                
                comparison_data = pd.DataFrame({
                    'Image': [r['image_name'] for r in results],
                    'Risk Score': [r['risk_score'] for r in results],
                    'Confidence': [r['confidence'] for r in results]
                })
                
                fig = px.bar(comparison_data, x='Image', y='Risk Score', 
                           color='Risk Score', color_continuous_scale="Reds",
                           title="ER Risk Score Comparison Across Images")
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence comparison
                fig2 = px.bar(comparison_data, x='Image', y='Confidence',
                            color='Confidence', color_continuous_scale="Blues",
                            title="Analysis Confidence Comparison")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Overall assessment
                st.subheader("🔍 Overall Assessment")
                
                avg_risk = np.mean([r['risk_score'] for r in results])
                avg_confidence = np.mean([r['confidence'] for r in results])
                high_risk_count = sum(1 for r in results if "High Risk" in r['risk_level'])
                
                assessment_col1, assessment_col2, assessment_col3 = st.columns(3)
                
                with assessment_col1:
                    st.metric("Average Risk Score", f"{avg_risk:.1f}%")
                
                with assessment_col2:
                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                
                with assessment_col3:
                    st.metric("High Risk Images", f"{high_risk_count}/{len(results)}")
                
                # Overall recommendation
                if avg_risk >= 60:
                    overall_risk = "High Risk"
                    risk_color = "red"
                elif avg_risk >= 30:
                    overall_risk = "Moderate Risk"
                    risk_color = "orange"
                else:
                    overall_risk = "Low Risk"
                    risk_color = "green"
                
                st.markdown(f"### Overall Assessment: :{risk_color}[{overall_risk}]")
                
                # Recommendations based on overall assessment
                recommendations = get_comprehensive_recommendations(
                    overall_risk, 
                    avg_confidence, 
                    {'risk_score': avg_risk, 'confidence': avg_confidence}
                )
                
                # Save best result (highest confidence)
                best_result = max(results, key=lambda x: x['confidence'])
                result_entry = {
                    'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'risk': overall_risk,
                    'score': f"{avg_risk:.1f}%",
                    'confidence': f"{avg_confidence:.1f}%",
                    'type': 'Multi-Image ER Analysis',
                    'er_results': best_result,
                    'total_images': len(results)
                }
                st.session_state.risk_history.append(result_entry)
                st.session_state.last_test_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                st.success(f"✅ Multi-image ER analysis complete! Analyzed {len(results)} images.")

elif page == get_text("tracker"):
    st.header("📈 Enhanced Progress Tracker")
    st.write("*Specialized tracking for ER+ breast cancer biomarkers*")
    
    if not st.session_state.risk_history:
        st.info("No data to display yet. Complete some assessments first!")
    else:
        # ER biomarker trends
        if st.session_state.biomarker_history:
            st.subheader("🧬 ER Biomarker Trends Over Time")
            
            df_bio = pd.DataFrame(st.session_state.biomarker_history)
            df_bio['date'] = pd.to_datetime(df_bio['date'])
            
            # ER-focused chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_bio['date'],
                y=df_bio['ER'],
                mode='lines+markers',
                name='ER Intensity',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10)
            ))
            
            # Add PR and HER2 traces only if columns exist
            if 'PR' in df_bio.columns:
                fig.add_trace(go.Scatter(
                    x=df_bio['date'],
                    y=df_bio['PR'],
                    mode='lines+markers',
                    name='PR Intensity',
                    line=dict(color='#4ECDC4', width=2),
                    marker=dict(size=8)
                ))
            
            if 'HER2' in df_bio.columns:
                fig.add_trace(go.Scatter(
                    x=df_bio['date'],
                    y=df_bio['HER2'],
                    mode='lines+markers',
                    name='HER2 Intensity',
                    line=dict(color='#45B7D1', width=2),
                    marker=dict(size=8)
                ))
            
            # Add threshold lines for ER
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="red",
                annotation_text="High Risk Threshold (50%)"
            )
            
            fig.add_hline(
                y=20,
                line_dash="dash",
                line_color="orange",
                annotation_text="Moderate Risk Threshold (20%)"
            )
            
            fig.update_layout(
                title="ER Intensity Tracking Over Time",
                xaxis_title="Date",
                yaxis_title="ER Intensity (%)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Alert if ER crosses threshold
            latest_bio = df_bio.iloc[-1]
            if latest_bio['ER'] > 50:
                st.error(f"⚠️ **High Risk Alert**: ER level is {latest_bio['ER']:.1f}% (above 50% threshold)")
            elif latest_bio['ER'] > 20:
                st.warning(f"⚡ **Moderate Risk Alert**: ER level is {latest_bio['ER']:.1f}% (above 20% threshold)")
            else:
                st.success(f"✅ **Low Risk**: ER level is {latest_bio['ER']:.1f}% (below risk thresholds)")
        
        # Overall risk timeline
        st.subheader("📊 Overall Risk Timeline")
        
        df_risk = pd.DataFrame(st.session_state.risk_history)
        df_risk['date'] = pd.to_datetime(df_risk['date'])
        
        # Risk level mapping
        risk_mapping = {"Low Risk": 1, "Moderate Risk": 2, "High Risk": 3}
        df_risk['risk_numeric'] = df_risk['risk'].map(risk_mapping)
        
        fig_risk = go.Figure()
        colors_risk = {'Low Risk': 'green', 'Moderate Risk': 'orange', 'High Risk': 'red'}
        
        for risk in df_risk['risk'].unique():
            if risk in colors_risk:
                risk_data = df_risk[df_risk['risk'] == risk]
                fig_risk.add_trace(go.Scatter(
                    x=risk_data['date'],
                    y=risk_data['risk_numeric'],
                    mode='markers+lines',
                    name=risk,
                    marker=dict(color=colors_risk[risk], size=10),
                    line=dict(color=colors_risk[risk])
                ))
        
        fig_risk.update_layout(
            title="Risk Level Progression",
            xaxis_title="Date",
            yaxis_title="Risk Level",
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3],
                ticktext=['Low Risk', 'Moderate Risk', 'High Risk']
            ),
            height=400
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Test adherence tracking
        st.subheader("📅 Test Adherence Tracking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            adherence_score = get_adherence_score()
            st.metric("Adherence Score", f"{adherence_score}%")
            
            if adherence_score >= 80:
                st.success("✅ Excellent adherence!")
            elif adherence_score >= 60:
                st.warning("⚡ Good adherence, but room for improvement")
            else:
                st.error("⚠️ Poor adherence - consider setting reminders")
        
        with col2:
            if st.session_state.last_test_date:
                last_test = datetime.datetime.strptime(st.session_state.last_test_date, "%Y-%m-%d")
                days_since = (datetime.datetime.now() - last_test).days
                next_due = last_test + datetime.timedelta(days=90)
                
                st.metric("Days Since Last Test", days_since)
                st.write(f"Next test due: {next_due.strftime('%Y-%m-%d')}")
        
        # Recent results table
        st.subheader("📋 Recent Test Results")
        display_df = df_risk[['date', 'risk', 'score', 'type']].sort_values('date', ascending=False).head(10)
        st.dataframe(display_df, use_container_width=True)

elif page == get_text("family"):
    st.header("👨‍👩‍👧‍👦 Enhanced Family History Assessment")
    st.write("*Specialized for ER+ breast cancer genetic risk factors*")
    
    # BRCA and genetic risk section
    st.subheader("🧬 Genetic Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First-degree relatives:**")
        mother_cancer = st.checkbox("Mother had breast/ovarian cancer")
        sister_cancer = st.checkbox("Sister(s) had breast/ovarian cancer")
        daughter_cancer = st.checkbox("Daughter(s) had breast/ovarian cancer")
        
        st.write("**Second-degree relatives:**")
        grandmother_cancer = st.checkbox("Grandmother had breast/ovarian cancer")
        aunt_cancer = st.checkbox("Aunt(s) had breast/ovarian cancer")
    
    with col2:
        st.write("**Genetic factors:**")
        brca_positive = st.checkbox("Known BRCA1/BRCA2 mutation in family")
        genetic_testing = st.checkbox("Family member had genetic testing")
        multiple_cancers = st.checkbox("Multiple cancers in same person")
        
        st.write("**Age factors:**")
        early_onset = st.checkbox("Cancer diagnosed before age 50")
        very_early = st.checkbox("Cancer diagnosed before age 40")
    
    # Additional ER+ specific factors
    st.subheader("🎯 ER+ Specific Risk Factors")
    
    col3, col4 = st.columns(2)
    
    with col3:
        hormone_therapy = st.checkbox("Family history of hormone therapy use")
        late_menopause = st.checkbox("Family history of late menopause (after 55)")
        no_pregnancies = st.checkbox("Family history of no pregnancies")
    
    with col4:
        dense_breasts = st.checkbox("Family history of dense breast tissue")
        hormone_positive = st.checkbox("Family history of hormone-positive cancers")
        lobular_cancer = st.checkbox("Family history of lobular carcinoma")
    
    if st.button("🔬 Calculate Comprehensive Risk", type="primary"):
        family_data = {
            'mother_cancer': mother_cancer,
            'sister_cancer': sister_cancer,
            'daughter_cancer': daughter_cancer,
            'grandmother_cancer': grandmother_cancer,
            'aunt_cancer': aunt_cancer,
            'brca_positive': brca_positive,
            'genetic_testing': genetic_testing,
            'multiple_cancers': multiple_cancers,
            'early_onset': early_onset,
            'very_early': very_early,
            'hormone_therapy': hormone_therapy,
            'late_menopause': late_menopause,
            'no_pregnancies': no_pregnancies,
            'dense_breasts': dense_breasts,
            'hormone_positive': hormone_positive,
            'lobular_cancer': lobular_cancer
        }
        
        st.session_state.family_history = family_data
        
        # Enhanced risk calculation
        risk_score = 0.1  # Base risk
        high_risk_factors = 0
        moderate_risk_factors = 0
        
        # High risk factors
        if brca_positive:
            risk_score += 0.5
            high_risk_factors += 1
        if mother_cancer:
            risk_score += 0.3
            high_risk_factors += 1
        if sister_cancer:
            risk_score += 0.25
            high_risk_factors += 1
        if very_early:
            risk_score += 0.3
            high_risk_factors += 1
        if multiple_cancers:
            risk_score += 0.2
            high_risk_factors += 1
        
        # Moderate risk factors
        if grandmother_cancer:
            risk_score += 0.15
            moderate_risk_factors += 1
        if aunt_cancer:
            risk_score += 0.1
            moderate_risk_factors += 1
        if early_onset:
            risk_score += 0.15
            moderate_risk_factors += 1
        if hormone_positive:
            risk_score += 0.1
            moderate_risk_factors += 1
        if dense_breasts:
            risk_score += 0.1
            moderate_risk_factors += 1
        
        risk_score = min(risk_score, 1.0)  # Cap at 100%
        
        # Display results
        st.subheader("🔍 Family Risk Assessment Results")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Risk Score", f"{risk_score*100:.1f}%")
        
        with col_b:
            st.metric("High Risk Factors", high_risk_factors)
        
        with col_c:
            st.metric("Moderate Risk Factors", moderate_risk_factors)
        
        # Risk level determination
        if risk_score > 0.6:
            st.error("⚠️ **High Genetic Risk**")
            st.write("**Recommendations:**")
            st.write("• Consider genetic counseling immediately")
            st.write("• Discuss BRCA testing with healthcare provider")
            st.write("• Consider enhanced screening (MRI + mammography)")
            st.write("• Discuss preventive options with oncologist")
        elif risk_score > 0.3:
            st.warning("⚡ **Moderate Genetic Risk**")
            st.write("**Recommendations:**")
            st.write("• Discuss family history with healthcare provider")
            st.write("• Consider genetic counseling")
            st.write("• Follow enhanced screening guidelines")
            st.write("• Maintain detailed family health records")
        else:
            st.success("✅ **Average Genetic Risk**")
            st.write("**Recommendations:**")
            st.write("• Continue routine screening")
            st.write("• Maintain healthy lifestyle")
            st.write("• Stay informed about family health changes")
        
        # Genetic counseling resources
        st.subheader("🧬 Genetic Counseling Resources")
        
        if high_risk_factors > 0 or risk_score > 0.5:
            st.write("**Recommended Genetic Counseling Centers:**")
            
            genetic_centers = [
                {"name": "Philippine Genome Center", "location": "UP Diliman", "phone": "(02) 8981-8500"},
                {"name": "St. Luke's Genetic Counseling", "location": "BGC/QC", "phone": "(02) 8789-7700"},
                {"name": "Makati Medical Center Genetics", "location": "Makati", "phone": "(02) 8888-8999"},
                {"name": "Asian Hospital Genetics", "location": "Muntinlupa", "phone": "(02) 8771-9000"}
            ]
            
            for center in genetic_centers:
                with st.expander(center["name"]):
                    st.write(f"**Location**: {center['location']}")
                    st.write(f"**Phone**: {center['phone']}")
                    st.write("**Services**: BRCA testing, genetic counseling, risk assessment")

elif page == get_text("symptoms"):
    st.header("🔍 ER+ Specific Symptom Analysis")
    st.write("*AI-powered symptom assessment focused on ER+ breast cancer indicators*")
    
    # Symptom categories
    st.subheader("Physical Symptoms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Breast Changes (Rate 0-5):**")
        lumps = st.slider("Unusual lumps or thickening", 0, 5, 0)
        skin_changes = st.slider("Skin texture changes (dimpling, puckering)", 0, 5, 0)
        nipple_discharge = st.slider("Nipple discharge (especially bloody)", 0, 5, 0)
        breast_pain = st.slider("Persistent breast pain", 0, 5, 0)
        size_changes = st.slider("Changes in breast size or shape", 0, 5, 0)
    
    with col2:
        st.write("**ER+ Specific Symptoms:**")
        nipple_inversion = st.slider("Nipple inversion or retraction", 0, 5, 0)
        skin_redness = st.slider("Skin redness or warmth", 0, 5, 0)
        lymph_nodes = st.slider("Swollen lymph nodes (armpit, collar)", 0, 5, 0)
        breast_heaviness = st.slider("Breast heaviness or fullness", 0, 5, 0)
        menstrual_changes = st.slider("Unusual menstrual changes", 0, 5, 0)
    
    # Duration and frequency
    st.subheader("Symptom Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        duration = st.selectbox(
            "How long have you experienced these symptoms?",
            ["Less than 1 week", "1-2 weeks", "2-4 weeks", "1-3 months", "More than 3 months"]
        )
        
        frequency = st.selectbox(
            "How often do you experience these symptoms?",
            ["Rarely", "Sometimes", "Often", "Daily", "Constantly"]
        )
    
    with col4:
        menstrual_relation = st.selectbox(
            "Are symptoms related to menstrual cycle?",
            ["No pattern", "Worse before period", "Worse during period", "Worse after period", "No relation"]
        )
        
        pain_type = st.selectbox(
            "If experiencing pain, what type?",
            ["No pain", "Sharp/stabbing", "Dull ache", "Burning", "Throbbing"]
        )
    
    # Lifestyle factors
    st.subheader("Lifestyle & Hormonal Factors")
    
    col5, col6 = st.columns(2)
    
    with col5:
        hormone_therapy = st.checkbox("Currently on hormone therapy")
        birth_control = st.checkbox("Currently using hormonal birth control")
        pregnancy_history = st.selectbox("Pregnancy history", ["Never pregnant", "1-2 pregnancies", "3+ pregnancies"])
    
    with col6:
        breastfeeding = st.checkbox("History of breastfeeding")
        menopause_status = st.selectbox("Menopause status", ["Pre-menopausal", "Peri-menopausal", "Post-menopausal"])
        family_history_input = st.checkbox("Family history of breast cancer")
    
    if st.button("🔬 Analyze Symptoms", type="primary"):
        # Enhanced symptom analysis
        symptoms_data = {
            'lumps': lumps,
            'skin_changes': skin_changes,
            'nipple_discharge': nipple_discharge,
            'breast_pain': breast_pain,
            'size_changes': size_changes,
            'nipple_inversion': nipple_inversion,
            'skin_redness': skin_redness,
            'lymph_nodes': lymph_nodes,
            'breast_heaviness': breast_heaviness,
            'menstrual_changes': menstrual_changes
        }
        
        # Save symptoms to session state
        st.session_state.symptoms = symptoms_data
        
        # Calculate risk with enhanced factors
        base_risk = calculate_symptom_risk_score(symptoms_data)
        
        # Adjust for duration and frequency
        duration_multiplier = {
            "Less than 1 week": 0.5,
            "1-2 weeks": 0.7,
            "2-4 weeks": 0.9,
            "1-3 months": 1.2,
            "More than 3 months": 1.5
        }
        
        frequency_multiplier = {
            "Rarely": 0.6,
            "Sometimes": 0.8,
            "Often": 1.0,
            "Daily": 1.3,
            "Constantly": 1.5
        }
        
        # Adjust for ER+ specific factors
        er_risk_multiplier = 1.0
        if hormone_therapy:
            er_risk_multiplier += 0.2
        if birth_control:
            er_risk_multiplier += 0.1
        if pregnancy_history == "Never pregnant":
            er_risk_multiplier += 0.1
        if not breastfeeding:
            er_risk_multiplier += 0.1
        if menopause_status == "Post-menopausal":
            er_risk_multiplier += 0.15
        
        # Final risk calculation
        adjusted_risk = base_risk * duration_multiplier[duration] * frequency_multiplier[frequency] * er_risk_multiplier
        adjusted_risk = min(adjusted_risk, 1.0)  # Cap at 100%
        
        # Determine risk level
        if adjusted_risk < 0.3:
            risk_level = "Low Risk"
            color = "green"
        elif adjusted_risk < 0.6:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        # Display results
        st.subheader("🎯 ER+ Symptom Analysis Results")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.markdown(f"**Risk Level**: :{color}[{risk_level}]")
        
        with col_result2:
            st.metric("Risk Score", f"{adjusted_risk*100:.1f}%")
        
        with col_result3:
            st.metric("Priority Level", "High" if adjusted_risk > 0.6 else "Medium" if adjusted_risk > 0.3 else "Low")
        
        # Detailed analysis
        st.subheader("📊 Detailed Symptom Analysis")
        
        # Symptom severity breakdown
        symptom_names = list(symptoms_data.keys())
        symptom_scores = list(symptoms_data.values())
        
        fig = px.bar(
            x=symptom_names,
            y=symptom_scores,
            title="Symptom Severity Breakdown",
            color=symptom_scores,
            color_continuous_scale="Reds"
        )
        fig.update_layout(xaxis_title="Symptoms", yaxis_title="Severity (0-5)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors contribution
        st.subheader("🔍 Risk Factor Contributions")
        
        risk_factors = {
            'Base Symptoms': base_risk * 100,
            'Duration Factor': (duration_multiplier[duration] - 1) * 100,
            'Frequency Factor': (frequency_multiplier[frequency] - 1) * 100,
            'ER+ Specific': (er_risk_multiplier - 1) * 100
        }
        
        factors_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Contribution'])
        fig2 = px.bar(factors_df, x='Factor', y='Contribution', title="Risk Factor Contributions (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Personalized recommendations
        st.subheader("🎯 Personalized Recommendations")
        
        if risk_level == "High Risk":
            st.error("⚠️ **Immediate Medical Attention Recommended**")
            st.write("**Urgent Actions:**")
            st.write("• Schedule appointment with healthcare provider within 1-2 weeks")
            st.write("• Document all symptoms with dates and severity")
            st.write("• Consider seeking second opinion if symptoms persist")
            st.write("• Avoid self-medication or delay in seeking care")
            
            if lymph_nodes > 3:
                st.write("• **Special attention**: Lymph node swelling requires immediate evaluation")
            if nipple_discharge > 3:
                st.write("• **Special attention**: Nipple discharge may require cytology testing")
        
        elif risk_level == "Moderate Risk":
            st.warning("⚡ **Medical Consultation Recommended**")
            st.write("**Recommended Actions:**")
            st.write("• Schedule appointment with healthcare provider within 2-4 weeks")
            st.write("• Monitor symptoms closely and document changes")
            st.write("• Continue monthly self-examinations")
            st.write("• Consider lifestyle modifications (diet, exercise)")
            
            if hormone_therapy:
                st.write("• **Note**: Discuss hormone therapy risks with provider")
        
        else:
            st.success("✅ **Continue Regular Monitoring**")
            st.write("**Recommended Actions:**")
            st.write("• Continue monthly self-examinations")
            st.write("• Maintain regular screening schedule")
            st.write("• Follow healthy lifestyle practices")
            st.write("• Stay aware of family history changes")
        
        # Save results
        result_entry = {
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'risk': risk_level,
            'score': f"{adjusted_risk*100:.1f}%",
            'type': 'ER+ Symptom Analysis',
            'symptom_details': symptoms_data
        }
        st.session_state.risk_history.append(result_entry)
        
        st.success("✅ Analysis complete! Results saved to your progress tracker.")

elif page == get_text("chat"):
    st.header("💬 ER+ Support Chat")
    st.write("*Specialized support for ER+ breast cancer concerns*")
    
    # Chat interface with ER+ specific responses
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
    
    # Quick topic buttons
    st.subheader("🎯 Quick Topics")
    
    topic_cols = st.columns(4)
    
    with topic_cols[0]:
        if st.button("💊 Hormone Therapy"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'Tell me about hormone therapy for ER+ breast cancer'
            })
    
    with topic_cols[1]:
        if st.button("🧬 ER+ Meaning"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'What does ER+ mean in breast cancer?'
            })
    
    with topic_cols[2]:
        if st.button("📊 Test Results"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'How to interpret my test results?'
            })
    
    with topic_cols[3]:
        if st.button("🏥 Next Steps"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'What should I do next?'
            })
    
    # Chat input
    user_input = st.chat_input("Ask me anything about ER+ breast cancer...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Enhanced AI responses for ER+ specific topics
        responses = {
            'hormone therapy': "ER+ breast cancer means your cancer cells have estrogen receptors. Hormone therapy (like tamoxifen or aromatase inhibitors) can block estrogen from fueling cancer growth. This is often very effective for ER+ cancers. Always discuss treatment options with your oncologist.",
            'er+': "ER+ (Estrogen Receptor Positive) means your cancer cells have receptors that bind to estrogen hormone. About 70% of breast cancers are ER+. The good news is that ER+ cancers often respond well to hormone therapy treatments.",
            'test results': "Test results showing ER+ status are important for treatment planning. Your pathology report will show the percentage of cells that are ER+. Higher percentages often indicate better response to hormone therapy. Share results with your healthcare team for personalized treatment planning.",
            'scared': "It's completely normal to feel anxious about ER+ breast cancer. Many people with ER+ cancer have excellent outcomes with proper treatment. You're taking positive steps by monitoring your health. Consider joining support groups and staying connected with your healthcare team.",
            'treatment': "ER+ breast cancer treatment often includes hormone therapy, which can be very effective. Treatment plans are personalized based on your specific situation. Common treatments include tamoxifen, aromatase inhibitors, and sometimes chemotherapy. Your oncologist will create the best plan for you.",
            'diet': "While there's no specific 'ER+ diet,' maintaining a healthy lifestyle is important. Some research suggests limiting alcohol and maintaining a healthy weight may be beneficial. Discuss any dietary supplements with your healthcare provider, as some may interact with hormone therapy.",
            'exercise': "Regular exercise can be beneficial for people with ER+ breast cancer. It may help reduce recurrence risk and improve overall health. Start slowly and gradually increase activity. Always consult your healthcare provider before starting new exercise programs.",
            'family': "Having ER+ breast cancer doesn't necessarily mean all family members will develop the same type. However, family history is important for risk assessment. Consider genetic counseling if you have strong family history of breast or ovarian cancer.",
            'default': "I understand your concern about ER+ breast cancer. While I can provide general information, please remember that this chat is for support only. For specific medical advice about ER+ breast cancer, always consult with your healthcare professional or oncologist."
        }
        
        # Enhanced keyword matching
        response = responses['default']
        user_lower = user_input.lower()
        
        for key in responses:
            if key in user_lower:
                response = responses[key]
                break
        
        # Add context based on user's data
        if st.session_state.risk_history:
            latest_risk = st.session_state.risk_history[-1]['risk']
            if latest_risk == "High Risk" and 'next' in user_lower:
                response += f"\n\nBased on your recent {latest_risk} assessment, I recommend scheduling an appointment with a healthcare provider as soon as possible for proper evaluation."
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()

elif page == get_text("resources"):
    st.header("📚 ER+ Breast Cancer Resources")
    st.write("*Specialized resources for Estrogen Receptor Positive breast cancer*")
    
    # Resource tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Specialized Centers", "📞 Support Lines", "🌐 ER+ Resources", "🧬 Research Centers"])
    
    with tab1:
        st.subheader("ER+ Specialized Treatment Centers")
        
        # Location-based clinic finder
        st.write("**Find Centers Near You:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_city = st.selectbox("Select Your City", ["Manila", "Quezon City", "Cebu", "Davao", "Other"])
        
        with col2:
            if user_city != "Other":
                barangay_options = {
                    "Manila": ["Ermita", "Malate", "Sampaloc"],
                    "Quezon City": ["Diliman", "Cubao", "Bago Bantay"],
                    "Cebu": ["Lahug", "Banilad", "Mabolo"],
                    "Davao": ["Poblacion", "Talomo", "Buhangin"]
                }
                user_barangay = st.selectbox("Select Barangay", 
                    barangay_options.get(user_city, ["No barangays available"])
                )
            else:
                user_barangay = None
        
        if st.button("🔍 Find Nearby Centers"):
            if user_city != "Other" and 'user_barangay' in locals():
                clinics = get_nearby_clinics(user_city, user_barangay)
                st.session_state.user_location = {"city": user_city, "barangay": user_barangay}
                
                if clinics:
                    st.success(f"Found {len(clinics)} centers in {user_barangay}, {user_city}:")
                    
                    for clinic in clinics:
                        with st.expander(f"📍 {clinic['name']}"):
                            st.write(f"**Phone**: {clinic['phone']}")
                            st.write(f"**Services**: {clinic['services']}")
                            st.write("**Specializations**: ER+ breast cancer treatment, hormone therapy")
                            
                            # Add distance and directions (simulated)
                            st.write(f"**Estimated Distance**: {np.random.randint(1, 10)} km")
                            st.write("**Transportation**: Jeepney, Bus, Taxi available")
                            
                            if st.button(f"📞 Call {clinic['name']}", key=f"call_{clinic['name']}"):
                                st.info(f"Call {clinic['phone']} to schedule an appointment")
                else:
                    st.warning("No specialized centers found in your area. Showing general recommendations:")
                    
                    general_centers = [
                        {"name": "Philippine General Hospital", "phone": "(02) 8554-8400", "specialization": "Comprehensive cancer care"},
                        {"name": "National Kidney and Transplant Institute", "phone": "(02) 8981-0300", "specialization": "Oncology services"},
                        {"name": "Philippine Heart Center", "phone": "(02) 8925-2401", "specialization": "Cardio-oncology"}
                    ]
                    
                    for center in general_centers:
                        with st.expander(center["name"]):
                            st.write(f"**Phone**: {center['phone']}")
                            st.write(f"**Specialization**: {center['specialization']}")
            else:
                st.error("Please select both city and barangay to find nearby centers.")
    
    with tab2:
        st.subheader("ER+ Support & Crisis Lines")
        
        # Emergency and support contacts
        support_lines = [
            {"name": "Philippine Cancer Society", "phone": "(02) 8927-2394", "hours": "24/7", "type": "General Support"},
            {"name": "Breast Cancer Support Philippines", "phone": "(02) 8426-7394", "hours": "9 AM - 5 PM", "type": "Peer Support"},
            {"name": "DOH Health Hotline", "phone": "1555", "hours": "24/7", "type": "Medical Information"},
            {"name": "Crisis and Suicide Prevention", "phone": "(02) 8893-7603", "hours": "24/7", "type": "Mental Health"},
            {"name": "ER+ Breast Cancer Helpline", "phone": "(02) 8555-ER-BC", "hours": "24/7", "type": "Specialized Support"}
        ]
        
        for line in support_lines:
            with st.expander(f"📞 {line['name']}"):
                st.write(f"**Phone**: {line['phone']}")
                st.write(f"**Hours**: {line['hours']}")
                st.write(f"**Type**: {line['type']}")
                
                if line['type'] == "Specialized Support":
                    st.write("**Services**: ER+ specific questions, treatment navigation, emotional support")
                elif line['type'] == "Peer Support":
                    st.write("**Services**: Connect with other ER+ breast cancer survivors")
        
        # Online support communities
        st.subheader("🌐 Online Support Communities")
        
        communities = [
            {"name": "ER+ Warriors Philippines", "platform": "Facebook", "members": "2,500+"},
            {"name": "Breast Cancer Support PH", "platform": "Telegram", "members": "1,200+"},
            {"name": "Pink Ribbon Sisters", "platform": "WhatsApp", "members": "800+"},
            {"name": "ER+ Survivors Network", "platform": "Discord", "members": "500+"}
        ]
        
        for community in communities:
            st.write(f"**{community['name']}** - {community['platform']} ({community['members']} members)")
    
    with tab3:
        st.subheader("ER+ Educational Resources")
        
        # Categorized resources
        resource_categories = {
            "🔬 Understanding ER+ Cancer": [
                {"title": "What is ER+ Breast Cancer?", "url": "https://www.cancer.org/cancer/breast-cancer/understanding-a-breast-cancer-diagnosis/breast-cancer-hormone-receptor-status.html"},
                {"title": "ER+ vs ER- Differences", "url": "https://breastcancer.org/symptoms/types/er-positive-pr-positive"},
                {"title": "How Hormone Therapy Works", "url": "https://www.mayoclinic.org/tests-procedures/hormone-therapy-for-breast-cancer/about/pac-20384943"}
            ],
            "💊 Treatment Options": [
                {"title": "Tamoxifen Information", "url": "https://www.cancer.org/cancer/breast-cancer/treatment/hormone-therapy/tamoxifen.html"},
                {"title": "Aromatase Inhibitors Guide", "url": "https://www.breastcancer.org/treatment/hormonal/aromatase-inhibitors"},
                {"title": "Side Effects Management", "url": "https://www.komen.org/breast-cancer/treatment/hormone-therapy/side-effects/"}
            ],
            "📊 Research & Statistics": [
                {"title": "ER+ Breast Cancer Statistics", "url": "https://www.cancer.org/cancer/breast-cancer/about/how-common-is-breast-cancer.html"},
                {"title": "Latest Research Findings", "url": "https://www.nature.com/subjects/breast-cancer"},
                {"title": "Clinical Trial Results", "url": "https://clinicaltrials.gov/ct2/results?cond=ER%2B+Breast+Cancer"}
            ],
            "🏠 Living with ER+ Cancer": [
                {"title": "Diet and Nutrition", "url": "https://www.cancer.org/treatment/survivorship-during-and-after-treatment/staying-active/nutrition.html"},
                {"title": "Exercise Guidelines", "url": "https://www.cancer.org/treatment/survivorship-during-and-after-treatment/staying-active/physical-activity-and-the-cancer-patient.html"},
                {"title": "Fertility and Pregnancy", "url": "https://www.cancer.org/cancer/breast-cancer/treatment/hormone-therapy/fertility-and-pregnancy.html"}
            ]
        }
        
        for category, resources in resource_categories.items():
            with st.expander(category):
                for resource in resources:
                    st.write(f"📄 [{resource['title']}]({resource['url']})")
    
    with tab4:
        st.subheader("ER+ Research Centers & Clinical Trials")
        
        # Research institutions
        research_centers = [
            {
                "name": "Philippine Genome Center",
                "location": "UP Diliman, Quezon City",
                "phone": "(02) 8981-8500",
                "research_focus": "Genetic markers in ER+ breast cancer, BRCA testing",
                "current_studies": "ER+ biomarker validation, personalized medicine"
            },
            {
                "name": "National Institute of Health",
                "location": "Manila",
                "phone": "(02) 8807-2628",
                "research_focus": "ER+ treatment protocols, hormone therapy optimization",
                "current_studies": "Tamoxifen vs AI effectiveness in Filipino population"
            },
            {
                "name": "St. Luke's Cancer Institute",
                "location": "BGC/Quezon City",
                "phone": "(02) 8789-7700",
                "research_focus": "ER+ cancer survivorship, quality of life studies",
                "current_studies": "Long-term effects of hormone therapy"
            }
        ]
        
        for center in research_centers:
            with st.expander(f"🔬 {center['name']}"):
                st.write(f"**Location**: {center['location']}")
                st.write(f"**Phone**: {center['phone']}")
                st.write(f"**Research Focus**: {center['research_focus']}")
                st.write(f"**Current Studies**: {center['current_studies']}")
                
                if st.button(f"Learn More About Studies", key=f"research_{center['name']}"):
                    st.info("Contact the center directly for information about participating in research studies.")
        
        # Current clinical trials (simulated)
        st.subheader("🧪 Current ER+ Clinical Trials")
        
        trials = [
            {
                "title": "ER+ Biomarker Validation Study",
                "phase": "Phase II",
                "location": "Multiple centers nationwide",
                "eligibility": "ER+ breast cancer patients, 21-70 years",
                "description": "Testing new biomarkers for ER+ cancer prognosis"
            },
            {
                "title": "Hormone Therapy Optimization Trial",
                "phase": "Phase III",
                "location": "Manila, Cebu, Davao",
                "eligibility": "Post-menopausal women with ER+ cancer",
                "description": "Comparing different hormone therapy regimens"
            },
            {
                "title": "ER+ Prevention Study",
                "phase": "Phase I",
                "location": "Philippine Genome Center",
                "eligibility": "High-risk women with family history",
                "description": "Testing preventive interventions for ER+ cancer"
            }
        ]
        
        for trial in trials:
            with st.expander(f"🧪 {trial['title']} - {trial['phase']}"):
                st.write(f"**Location**: {trial['location']}")
                st.write(f"**Eligibility**: {trial['eligibility']}")
                st.write(f"**Description**: {trial['description']}")
                
                if st.button(f"Check Eligibility", key=f"trial_{trial['title']}"):
                    st.info("Contact your healthcare provider to discuss clinical trial participation.")

elif page == get_text("education"):
    st.header("🎓 ER+ Breast Cancer Education Center")
    st.write("*Comprehensive education about Estrogen Receptor Positive breast cancer*")
    
    # Educational tabs
    edu_tabs = st.tabs(["🔬 ER+ Basics", "🧬 Biomarkers", "💊 Treatments", "📊 Statistics", "🏠 Self-Care"])
    
    with edu_tabs[0]:
        st.subheader("Understanding ER+ Breast Cancer")
        
        # Interactive learning modules
        with st.expander("🎯 What is ER+ Breast Cancer?"):
            st.write("""
            **Estrogen Receptor Positive (ER+) breast cancer** is the most common type of breast cancer, accounting for about 70% of all cases.
            
            **Key Points:**
            - Cancer cells have receptors that bind to estrogen hormone
            - Estrogen can fuel the growth of these cancer cells
            - Generally has better prognosis than ER- cancers
            - Responds well to hormone therapy treatments
            """)
            
            # Interactive diagram description
            st.info("📊 **ER+ Cancer Cell Diagram**: Cancer cells with estrogen receptors that can bind to estrogen hormones, potentially fueling cell growth.")
        
        with st.expander("🔄 How ER+ Cancer Develops"):
            st.write("""
            **The Process:**
            1. **Normal cells** have estrogen receptors for normal functions
            2. **DNA changes** occur in breast cells
            3. **Abnormal growth** begins when estrogen binds to receptors
            4. **Cancer cells multiply** fueled by estrogen
            5. **Tumor formation** occurs over time
            """)
            
            # Timeline visualization
            timeline_data = pd.DataFrame({
                'Stage': ['Normal Cell', 'DNA Change', 'Abnormal Growth', 'Cancer Cells', 'Tumor'],
                'Time': [0, 1, 2, 3, 4],
                'Risk': [0, 20, 40, 70, 100]
            })
            
            fig = px.line(timeline_data, x='Time', y='Risk', markers=True, 
                         title="ER+ Cancer Development Timeline")
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🎯 ER+ vs ER- Comparison"):
            comparison_data = {
                'Factor': ['Prevalence', 'Prognosis', 'Treatment Response', 'Growth Rate', 'Recurrence Risk'],
                'ER+': ['70%', 'Generally Better', 'Excellent with Hormones', 'Slower', 'Lower (with treatment)'],
                'ER-': ['30%', 'More Aggressive', 'Chemotherapy Focus', 'Faster', 'Higher']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
    
    with edu_tabs[1]:
        st.subheader("🧬 Biomarkers in ER+ Cancer")
        
        # Biomarker education
        biomarker_info = {
            "ER (Estrogen Receptor)": {
                "description": "Protein that binds to estrogen hormone",
                "normal_range": "0-10%",
                "positive_range": ">10%",
                "treatment_impact": "Responds to hormone therapy like tamoxifen",
                "importance": "Primary target for ER+ treatment"
            },
            "PR (Progesterone Receptor)": {
                "description": "Protein that binds to progesterone hormone",
                "normal_range": "0-10%",
                "positive_range": ">10%",
                "treatment_impact": "Often positive with ER, better prognosis",
                "importance": "Indicates hormone sensitivity"
            },
            "HER2": {
                "description": "Protein that promotes cell growth",
                "normal_range": "0-2+",
                "positive_range": "3+ or amplified",
                "treatment_impact": "Responds to targeted therapy like trastuzumab",
                "importance": "Important for treatment selection"
            },
            "Ki-67": {
                "description": "Protein present during cell division",
                "normal_range": "<15%",
                "positive_range": ">15%",
                "treatment_impact": "Higher levels may indicate need for chemotherapy",
                "importance": "Measures how fast cancer is growing"
            }
        }
        
        for biomarker, info in biomarker_info.items():
            with st.expander(f"🔬 {biomarker} Explained"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description**: {info['description']}")
                    st.write(f"**Normal Range**: {info['normal_range']}")
                    st.write(f"**Positive Range**: {info['positive_range']}")
                
                with col2:
                    st.write(f"**Treatment Impact**: {info['treatment_impact']}")
                    st.write(f"**Clinical Importance**: {info['importance']}")
                
                # Simulated biomarker visualization
                if biomarker == "ER (Estrogen Receptor)":
                    sample_data = pd.DataFrame({
                        'Patient': ['A', 'B', 'C', 'D', 'E'],
                        'ER%': [85, 70, 15, 5, 90]
                    })
                    fig = px.bar(sample_data, x='Patient', y='ER%', 
                               title="Sample ER Expression Levels")
                    fig.add_hline(y=10, line_dash="dash", line_color="red", 
                                annotation_text="ER+ Threshold (10%)")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Biomarker interpretation guide
        st.subheader("📊 Interpreting Your Biomarker Results")
        
        interpretation_guide = pd.DataFrame({
            'Biomarker Combination': [
                'ER+ PR+ HER2-',
                'ER+ PR+ HER2+',
                'ER+ PR- HER2-',
                'ER+ PR- HER2+',
                'ER- PR- HER2-',
                'ER- PR- HER2+'
            ],
            'Subtype': [
                'Luminal A-like',
                'Luminal B-like',
                'Luminal A-like',
                'Luminal B-like',
                'Triple Negative',
                'HER2-enriched'
            ],
            'Prognosis': [
                'Excellent',
                'Good',
                'Good',
                'Good',
                'Variable',
                'Good with treatment'
            ],
            'Primary Treatment': [
                'Hormone Therapy',
                'Hormone + Targeted',
                'Hormone Therapy',
                'Hormone + Targeted',
                'Chemotherapy',
                'Targeted Therapy'
            ]
        })
        
        st.dataframe(interpretation_guide, use_container_width=True)
    
    with edu_tabs[2]:
        st.subheader("💊 ER+ Treatment Options")
        
        # Treatment categories
        treatment_tabs = st.tabs(["🌿 Hormone Therapy", "⚕️ Targeted Therapy", "💉 Chemotherapy", "🔬 Combination"])
        
        with treatment_tabs[0]:
            st.write("**Hormone Therapy - First Line for ER+ Cancer**")
            
            hormone_treatments = {
                "Tamoxifen": {
                    "mechanism": "Blocks estrogen receptors",
                    "best_for": "Pre-menopausal women",
                    "duration": "5-10 years",
                    "side_effects": "Hot flashes, blood clots (rare)",
                    "effectiveness": "Reduces recurrence by 40-50%"
                },
                "Aromatase Inhibitors": {
                    "mechanism": "Blocks estrogen production",
                    "best_for": "Post-menopausal women",
                    "duration": "5-10 years",
                    "side_effects": "Joint pain, bone loss",
                    "effectiveness": "Slightly better than tamoxifen"
                },
                "Fulvestrant": {
                    "mechanism": "Destroys estrogen receptors",
                    "best_for": "Advanced/metastatic ER+ cancer",
                    "duration": "Until progression",
                    "side_effects": "Injection site reactions",
                    "effectiveness": "Effective for advanced disease"
                }
            }
            
            for treatment, details in hormone_treatments.items():
                with st.expander(f"💊 {treatment}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Mechanism**: {details['mechanism']}")
                        st.write(f"**Best For**: {details['best_for']}")
                        st.write(f"**Duration**: {details['duration']}")
                    
                    with col2:
                        st.write(f"**Side Effects**: {details['side_effects']}")
                        st.write(f"**Effectiveness**: {details['effectiveness']}")
        
        with treatment_tabs[1]:
            st.write("**Targeted Therapy - Precision Medicine**")
            
            targeted_treatments = {
                "CDK4/6 Inhibitors": {
                    "drugs": "Palbociclib, Ribociclib, Abemaciclib",
                    "combination": "With hormone therapy",
                    "benefit": "Delays disease progression",
                    "side_effects": "Low blood counts, fatigue"
                },
                "mTOR Inhibitors": {
                    "drugs": "Everolimus",
                    "combination": "With exemestane",
                    "benefit": "Overcomes hormone resistance",
                    "side_effects": "Lung inflammation, mouth sores"
                },
                "PIK3CA Inhibitors": {
                    "drugs": "Alpelisib",
                    "combination": "With fulvestrant",
                    "benefit": "For PIK3CA mutated tumors",
                    "side_effects": "High blood sugar, diarrhea"
                }
            }
            
            for treatment, details in targeted_treatments.items():
                with st.expander(f"🎯 {treatment}"):
                    st.write(f"**Drugs**: {details['drugs']}")
                    st.write(f"**Used With**: {details['combination']}")
                    st.write(f"**Benefit**: {details['benefit']}")
                    st.write(f"**Side Effects**: {details['side_effects']}")
        
        with treatment_tabs[2]:
            st.write("**Chemotherapy - When Needed**")
            
            st.write("""
            **Chemotherapy for ER+ Cancer:**
            - Usually reserved for high-risk cases
            - May be used if hormone therapy fails
            - Often combined with hormone therapy
            - Decision based on tumor characteristics
            """)
            
            chemo_indications = pd.DataFrame({
                'Indication': [
                    'Large tumor size (>5cm)',
                    'Lymph node involvement',
                    'High Ki-67 (>30%)',
                    'Grade 3 tumor',
                    'Hormone therapy resistance'
                ],
                'Likelihood of Chemo': [
                    'High',
                    'Moderate-High',
                    'Moderate',
                    'Moderate',
                    'High'
                ],
                'Rationale': [
                    'Size indicates aggressive disease',
                    'Spread to lymph nodes',
                    'Fast-growing tumor',
                    'Poorly differentiated cells',
                    'Need alternative treatment'
                ]
            })
            
            st.dataframe(chemo_indications, use_container_width=True)
        
        with treatment_tabs[3]:
            st.write("**Combination Therapy - Maximizing Effectiveness**")
            
            # Treatment algorithm flowchart (simulated)
            st.write("**Treatment Decision Algorithm:**")
            
            decision_tree = """
            ```
            ER+ Breast Cancer Diagnosis
            ↓
            Assess Risk Factors
            ├── Low Risk → Hormone Therapy Alone
            ├── Intermediate Risk → Hormone Therapy ± Chemotherapy
            └── High Risk → Chemotherapy + Hormone Therapy
            
            Monitor Response
            ├── Good Response → Continue
            ├── Partial Response → Add Targeted Therapy
            └── Progression → Switch to Different Combination
            ```
            """
            st.code(decision_tree)
    
    with edu_tabs[3]:
        st.subheader("📊 ER+ Breast Cancer Statistics")
        
        # Statistical visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival rates
            survival_data = pd.DataFrame({
                'Year': [1, 2, 3, 4, 5, 10],
                'ER+ Survival Rate': [95, 92, 88, 85, 82, 75],
                'Overall Survival Rate': [90, 85, 80, 75, 70, 60]
            })
            
            fig = px.line(survival_data, x='Year', y=['ER+ Survival Rate', 'Overall Survival Rate'],
                         title="ER+ vs Overall Survival Rates")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Treatment response rates
            response_data = pd.DataFrame({
                'Treatment': ['Hormone Therapy', 'Targeted Therapy', 'Chemotherapy', 'Combination'],
                'Response Rate': [70, 85, 60, 90]
            })
            
            fig = px.bar(response_data, x='Treatment', y='Response Rate',
                        title="Treatment Response Rates in ER+ Cancer")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key statistics
        st.subheader("🔢 Key ER+ Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric("Prevalence", "70%", "of breast cancers")
        
        with stats_cols[1]:
            st.metric("5-Year Survival", "85%", "with treatment")
        
        with stats_cols[2]:
            st.metric("Recurrence Rate", "15%", "in 10 years")
        
        with stats_cols[3]:
            st.metric("Treatment Response", "80%", "to hormone therapy")
        
        # Philippine statistics
        st.subheader("🇵🇭 Philippines ER+ Statistics")
        
        ph_stats = pd.DataFrame({
            'Metric': [
                'Annual ER+ Cases',
                'Average Age at Diagnosis',
                'Stage I-II Diagnosis Rate',
                'Access to Hormone Therapy',
                'Genetic Testing Availability'
            ],
            'Value': [
                '~12,000',
                '52 years',
                '65%',
                '70%',
                '20%'
            ],
            'Trend': [
                'Increasing',
                'Stable',
                'Improving',
                'Improving',
                'Improving'
            ]
        })
        
        st.dataframe(ph_stats, use_container_width=True)
    
    with edu_tabs[4]:
        st.subheader("🏠 Self-Care for ER+ Patients")
        
        # Self-care categories
        selfcare_tabs = st.tabs(["🍎 Nutrition", "🏃‍♀️ Exercise", "🧘‍♀️ Mental Health", "💊 Medication"])
        
        with selfcare_tabs[0]:
            st.write("**Nutrition Guidelines for ER+ Patients**")
            
            # Dietary recommendations
            diet_recommendations = {
                "🥬 Recommended Foods": [
                    "Cruciferous vegetables (broccoli, cauliflower)",
                    "Leafy greens (spinach, kale)",
                    "Berries and antioxidant-rich fruits",
                    "Whole grains and fiber",
                    "Lean proteins (fish, poultry)",
                    "Healthy fats (olive oil, avocados)"
                ],
                "⚠️ Foods to Limit": [
                    "Processed meats",
                    "High-fat dairy products",
                    "Refined sugars and sweets",
                    "Alcohol (discuss with doctor)",
                    "Highly processed foods",
                    "Excessive red meat"
                ],
                "🌿 Supplements to Discuss": [
                    "Vitamin D (bone health)",
                    "Calcium (with AI therapy)",
                    "Omega-3 fatty acids",
                    "Probiotics",
                    "Avoid: High-dose soy isoflavones",
                    "Avoid: Concentrated phytoestrogens"
                ]
            }
            
            for category, items in diet_recommendations.items():
                with st.expander(category):
                    for item in items:
                        st.write(f"• {item}")
        
        with selfcare_tabs[1]:
            st.write("**Exercise Guidelines for ER+ Patients**")
            
            # Exercise recommendations
            exercise_plan = {
                "🏃‍♀️ Aerobic Exercise": {
                    "frequency": "150 minutes/week moderate intensity",
                    "examples": "Walking, swimming, cycling",
                    "benefits": "Improves survival, reduces fatigue",
                    "precautions": "Start slowly, listen to body"
                },
                "💪 Strength Training": {
                    "frequency": "2-3 times/week",
                    "examples": "Resistance bands, light weights",
                    "benefits": "Maintains muscle mass, bone health",
                    "precautions": "Avoid heavy lifting if lymphedema risk"
                },
                "🧘‍♀️ Flexibility/Balance": {
                    "frequency": "Daily",
                    "examples": "Yoga, stretching, tai chi",
                    "benefits": "Reduces stress, improves quality of life",
                    "precautions": "Modified poses if needed"
                }
            }
            
            for exercise_type, details in exercise_plan.items():
                with st.expander(exercise_type):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Frequency**: {details['frequency']}")
                        st.write(f"**Examples**: {details['examples']}")
                    with col2:
                        st.write(f"**Benefits**: {details['benefits']}")
                        st.write(f"**Precautions**: {details['precautions']}")
        
        with selfcare_tabs[2]:
            st.write("**Mental Health Support**")
            
            # Mental health resources
            mental_health_strategies = {
                "🧠 Coping Strategies": [
                    "Mindfulness meditation",
                    "Deep breathing exercises",
                    "Journaling thoughts and feelings",
                    "Connecting with support groups",
                    "Maintaining social connections",
                    "Engaging in hobbies"
                ],
                "⚠️ Warning Signs": [
                    "Persistent sadness or anxiety",
                    "Loss of interest in activities",
                    "Sleep disturbances",
                    "Appetite changes",
                    "Difficulty concentrating",
                    "Thoughts of self-harm"
                ],
                "🆘 When to Seek Help": [
                    "Symptoms interfere with daily life",
                    "Feeling overwhelmed consistently",
                    "Relationship problems",
                    "Substance use concerns",
                    "Persistent fatigue",
                    "Any concerning symptoms"
                ]
            }
            
            for category, items in mental_health_strategies.items():
                with st.expander(category):
                    for item in items:
                        st.write(f"• {item}")
        
        with selfcare_tabs[3]:
            st.write("**Medication Management**")
            
            # Medication adherence tips
            med_management = {
                "💊 Adherence Tips": [
                    "Take medication at same time daily",
                    "Use pill organizers or apps",
                    "Set phone reminders",
                    "Connect with meal times",
                    "Keep medications visible",
                    "Discuss barriers with healthcare team"
                ],
                "📝 Tracking Side Effects": [
                    "Keep a daily symptom diary",
                    "Rate severity 1-10",
                    "Note timing and triggers",
                    "Document impact on activities",
                    "Share with healthcare team",
                    "Don't stop medications without consulting"
                ],
                "⚠️ Important Interactions": [
                    "Inform all doctors of ER+ treatment",
                    "Check with pharmacist before new medications",
                    "Discuss supplements and herbs",
                    "Be cautious with over-the-counter drugs",
                    "Avoid grapefruit with certain medications",
                    "Report any unusual symptoms"
                ]
            }
            
            for category, items in med_management.items():
                with st.expander(category):
                    for item in items:
                        st.write(f"• {item}")

elif page == get_text("export"):
    st.header("📤 Comprehensive Data Export")
    st.write("*Export your complete ER+ breast cancer risk monitoring data*")
    
    if not st.session_state.risk_history and not st.session_state.family_history:
        st.info("No data to export yet. Complete some assessments first!")
    else:
        # Export options
        export_tabs = st.tabs(["📄 PDF Report", "📊 Data Files", "🔄 Backup", "📧 Share"])
        
        with export_tabs[0]:
            st.subheader("📄 Generate PDF Health Report")
            
            # Report customization
            col1, col2 = st.columns(2)
            
            with col1:
                include_biomarkers = st.checkbox("Include Biomarker History", value=True)
                include_symptoms = st.checkbox("Include Symptom Analysis", value=True)
                include_family = st.checkbox("Include Family History", value=True)
            
            with col2:
                include_recommendations = st.checkbox("Include Recommendations", value=True)
                include_charts = st.checkbox("Include Progress Charts", value=True)
                include_resources = st.checkbox("Include Resource Links", value=True)
            
            if st.button("🔄 Generate PDF Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    try:
                        report_buffer = generate_pdf_report()
                        
                        if REPORTLAB_AVAILABLE:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=report_buffer,
                                file_name=f"ER+_breast_cancer_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                            st.success("✅ PDF report generated successfully!")
                        else:
                            st.download_button(
                                label="📄 Download Text Report",
                                data=report_buffer,
                                file_name=f"ER+_breast_cancer_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            st.warning("⚠️ PDF library not available. Generated text report instead.")
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.info("💡 Try refreshing the page and generating the report again.")
        
        with export_tabs[1]:
            st.subheader("📊 Export Data Files")
            
            # Prepare comprehensive export data
            export_data = {
                'user_profile': {
                    'user_id': f"er_plus_user_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'export_date': datetime.datetime.now().isoformat(),
                    'app_version': "ER+ Monitor v2.0",
                    'language': st.session_state.language
                },
                'risk_assessments': st.session_state.risk_history,
                'biomarker_history': st.session_state.biomarker_history,
                'family_history': st.session_state.family_history,
                'symptoms_history': st.session_state.symptoms,
                'chat_history': st.session_state.chat_history,
                'location_data': st.session_state.user_location,
                'adherence_metrics': {
                    'total_tests': len(st.session_state.risk_history),
                    'adherence_score': get_adherence_score(),
                    'last_test_date': st.session_state.last_test_date
                },
                'disclaimer': 'This data is for personal health tracking only and does not constitute medical advice. Always consult healthcare professionals for medical decisions.'
            }
            
            # JSON export
            json_data = json.dumps(export_data, indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download Complete Data (JSON)",
                    data=json_data,
                    file_name=f"er_plus_complete_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export for risk history
                if st.session_state.risk_history:
                    df_risk = pd.DataFrame(st.session_state.risk_history)
                    csv_data = df_risk.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Risk History (CSV)",
                        data=csv_data,
                        file_name=f"er_plus_risk_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Biomarker data export
            if st.session_state.biomarker_history:
                df_bio = pd.DataFrame(st.session_state.biomarker_history)
                bio_csv = df_bio.to_csv(index=False)
                
                st.download_button(
                    label="🧬 Download Biomarker Data (CSV)",
                    data=bio_csv,
                    file_name=f"er_plus_biomarkers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with export_tabs[2]:
            st.subheader("🔄 Data Backup & Restore")
            
            # Backup current data
            if st.button("💾 Create Backup"):
                backup_data = {
                    'backup_date': datetime.datetime.now().isoformat(),
                    'session_state': {
                        'risk_history': st.session_state.risk_history,
                        'biomarker_history': st.session_state.biomarker_history,
                        'family_history': st.session_state.family_history,
                        'symptoms': st.session_state.symptoms,
                        'last_test_date': st.session_state.last_test_date,
                        'user_location': st.session_state.user_location
                    }
                }
                
                backup_json = json.dumps(backup_data, indent=2)
                
                st.download_button(
                    label="💾 Download Backup File",
                    data=backup_json,
                    file_name=f"er_plus_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.success("✅ Backup created successfully!")
            
            # Restore from backup
            st.write("**Restore from Backup:**")
            restore_file = st.file_uploader("Upload Backup File", type=['json'])
            
            if restore_file:
                try:
                    backup_data = json.load(restore_file)
                    
                    if st.button("🔄 Restore Data", type="secondary"):
                        # Restore session state
                        session_data = backup_data['session_state']
                        
                        st.session_state.risk_history = session_data.get('risk_history', [])
                        st.session_state.biomarker_history = session_data.get('biomarker_history', [])
                        st.session_state.family_history = session_data.get('family_history', {})
                        st.session_state.symptoms = session_data.get('symptoms', [])
                        st.session_state.last_test_date = session_data.get('last_test_date', None)
                        st.session_state.user_location = session_data.get('user_location', {"city": "", "barangay": ""})
                        
                        st.success("✅ Data restored successfully!")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error restoring backup: {str(e)}")
        
        with export_tabs[3]:
            st.subheader("📧 Share Data with Healthcare Provider")
            
            # Generate shareable summary
            st.write("**Generate Summary for Healthcare Provider:**")
            
            # Summary options
            summary_options = {
                "Latest Risk Assessment": st.checkbox("Include latest risk assessment", value=True),
                "Biomarker Trends": st.checkbox("Include biomarker trends", value=True),
                "Family History": st.checkbox("Include family history", value=True),
                "Symptom Analysis": st.checkbox("Include symptom analysis", value=True),
                "Adherence Score": st.checkbox("Include test adherence", value=True)
            }
            
            if st.button("📋 Generate Healthcare Summary"):
                # Create medical summary
                medical_summary = f"""
                ER+ BREAST CANCER RISK MONITORING SUMMARY
                Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
                
                PATIENT OVERVIEW:
                - Total Risk Assessments: {len(st.session_state.risk_history)}
                - Test Adherence Score: {get_adherence_score()}%
                - Last Assessment Date: {st.session_state.last_test_date or 'N/A'}
                
                LATEST RISK ASSESSMENT:
                """
                
                if st.session_state.risk_history:
                    latest = st.session_state.risk_history[-1]
                    medical_summary += f"""
                - Risk Level: {latest['risk']}
                - Risk Score: {latest['score']}
                - Assessment Type: {latest['type']}
                - Date: {latest['date']}
                """
                
                if st.session_state.biomarker_history and summary_options["Biomarker Trends"]:
                    latest_bio = st.session_state.biomarker_history[-1]
                    medical_summary += f"""
                
                LATEST BIOMARKER LEVELS:
                - ER Intensity: {latest_bio['ER']:.1f}%
                - PR Intensity: {latest_bio.get('PR', 0):.1f}%
                - HER2 Intensity: {latest_bio.get('HER2', 0):.1f}%
                """
                
                if st.session_state.family_history and summary_options["Family History"]:
                    family_factors = [k for k, v in st.session_state.family_history.items() if v]
                    medical_summary += f"""
                
                FAMILY HISTORY RISK FACTORS:
                {chr(10).join(f"- {factor.replace('_', ' ').title()}" for factor in family_factors)}
                """
                
                medical_summary += """
                
                DISCLAIMER:
                This summary is generated from a patient self-monitoring app and is for 
                informational purposes only. Clinical correlation and professional medical 
                assessment are required for any medical decisions.
                
                App: ER+ Breast Cancer Risk Monitor v2.0
                """
                
                st.text_area("Medical Summary", medical_summary, height=400)
                
                st.download_button(
                    label="📄 Download Medical Summary",
                    data=medical_summary,
                    file_name=f"er_plus_medical_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        # Summary dashboard
        st.subheader("📊 Export Summary Dashboard")
        
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            st.metric("Total Assessments", len(st.session_state.risk_history))
        
        with summary_cols[1]:
            st.metric("Biomarker Tests", len(st.session_state.biomarker_history))
        
        with summary_cols[2]:
            family_risk_factors = sum(1 for v in st.session_state.family_history.values() if v) if st.session_state.family_history else 0
            st.metric("Family Risk Factors", family_risk_factors)
        
        with summary_cols[3]:
            adherence_score = get_adherence_score()
            st.metric("Adherence Score", f"{adherence_score}%")
        
        # Data completeness indicator
        st.subheader("📈 Data Completeness")
        
        completeness_data = {
            'Category': ['Risk Assessments', 'Biomarker Data', 'Family History', 'Symptoms', 'Location'],
            'Status': [
                'Complete' if st.session_state.risk_history else 'Incomplete',
                'Complete' if st.session_state.biomarker_history else 'Incomplete',
                'Complete' if st.session_state.family_history else 'Incomplete',
                'Complete' if st.session_state.symptoms else 'Incomplete',
                'Complete' if st.session_state.user_location['city'] else 'Incomplete'
            ],
            'Count': [
                len(st.session_state.risk_history),
                len(st.session_state.biomarker_history),
                sum(1 for v in st.session_state.family_history.values() if v) if st.session_state.family_history else 0,
                len(st.session_state.symptoms),
                1 if st.session_state.user_location['city'] else 0
            ]
        }
        
        completeness_df = pd.DataFrame(completeness_data)
        
        # Color code the status
        def color_status(val):
            color = 'green' if val == 'Complete' else 'red'
            return f'background-color: {color}; color: white'
        
        styled_df = completeness_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🎗️ **ER+ Breast Cancer Monitor v2.0**")
st.sidebar.markdown("💡 **Remember**: This app is for educational purposes only. Always consult healthcare professionals for medical advice.")
st.sidebar.markdown("🇵🇭 Made with ❤️ for Filipino communities")
st.sidebar.markdown("🔬 Specialized for ER+ breast cancer")

# Quick actions in sidebar
if st.sidebar.button("🚨 Emergency Contacts"):
    st.sidebar.write("**Emergency Hotlines:**")
    st.sidebar.write("DOH: 1555")
    st.sidebar.write("Emergency: 911")
    st.sidebar.write("Cancer Society: (02) 8927-2394")

if st.sidebar.button("📱 Quick Test Reminder"):
    if st.session_state.last_test_date:
        last_test = datetime.datetime.strptime(st.session_state.last_test_date, "%Y-%m-%d")
        days_since = (datetime.datetime.now() - last_test).days
        st.sidebar.write(f"Last test: {days_since} days ago")
        
        if days_since >= 90:
            st.sidebar.error("⏰ Test overdue!")
        else:
            st.sidebar.success("✅ On schedule")
    else:
        st.sidebar.info("No previous tests recorded")
import sqlite3

# Connect to database (file will be created)
conn = sqlite3.connect("patients.db")
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    result TEXT
)
''')

# Insert a result
cursor.execute("INSERT INTO predictions (patient_name, result) VALUES (?, ?)", 
               ("Patient 1", "HER2+ Luminal B"))
conn.commit()

# Show all predictions
cursor.execute("SELECT * FROM predictions")
print("Stored predictions:")
for row in cursor.fetchall():
    print(row)

conn.close()

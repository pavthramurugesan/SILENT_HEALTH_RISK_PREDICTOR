"""
╔══════════════════════════════════════════════════════════════╗
║         SILENT HEALTH RISK PREDICTOR — STREAMLIT APP         ║
║                                                              ║
║  HOW TO RUN:                                                 ║
║  1. Make sure you have run ALL cells in your Colab notebook  ║
║  2. Save your model files (Cell 10 in notebook)              ║
║  3. pip install streamlit                                     ║
║  4. streamlit run health_risk_app.py                         ║
║                                                              ║
║  FILES NEEDED in same folder as this script:                 ║
║    - best_risk_model.pkl                                     ║
║    - scaler.pkl                                              ║
║    - label_encoder.pkl                                       ║
║    - feature_names.pkl                                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Silent Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f1117; }

/* Header */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 50%, #1a1f2e 100%);
    border: 1px solid #2a3142;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #e2e8f0;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1rem;
    color: #718096;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2433;
}

/* Card */
.card {
    background: #1a1f2e;
    border: 1px solid #2a3142;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result cards */
.result-low {
    background: linear-gradient(135deg, #0d2418 0%, #1a1f2e 100%);
    border: 1px solid #276749;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-moderate {
    background: linear-gradient(135deg, #2d2106 0%, #1a1f2e 100%);
    border: 1px solid #b7791f;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, #2d0f0f 0%, #1a1f2e 100%);
    border: 1px solid #c53030;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.risk-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.risk-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    margin: 0.3rem 0;
    line-height: 1;
}
.risk-low  { color: #68d391; }
.risk-mod  { color: #f6ad55; }
.risk-high { color: #fc8181; }

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #a0aec0;
    width: 80px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #2d3748;
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill-low  { height: 100%; background: #68d391; border-radius: 4px; transition: width 0.8s ease; }
.prob-bar-fill-mod  { height: 100%; background: #f6ad55; border-radius: 4px; transition: width 0.8s ease; }
.prob-bar-fill-high { height: 100%; background: #fc8181; border-radius: 4px; transition: width 0.8s ease; }
.prob-pct {
    font-size: 0.85rem;
    font-weight: 600;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}

/* Metric pills */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 1rem;
}
.metric-pill {
    background: #0f1117;
    border: 1px solid #2a3142;
    border-radius: 10px;
    padding: 10px 14px;
}
.metric-pill-label {
    font-size: 0.65rem;
    color: #4a5568;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-pill-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 2px;
}

/* Suggestion box */
.suggestion-box {
    background: #0f1117;
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.suggestion-box p {
    color: #a0aec0;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

/* Streamlit overrides */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background-color: #1a1f2e !important;
    border: 1px solid #2a3142 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] > div {
    background-color: #1a1f2e !important;
    border: 1px solid #2a3142 !important;
    border-radius: 8px !important;
}
.stSlider > div > div {
    background-color: #2a3142 !important;
}
label, .stRadio label, .stCheckbox label {
    color: #a0aec0 !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    margin-top: 1rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    required = ['best_risk_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'feature_names.pkl']
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model    = joblib.load('best_risk_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    risk_le  = joblib.load('label_encoder.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, risk_le, features, []

model, scaler, risk_le, all_features, missing_files = load_models()


# ── Hero header ───────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Health Assessment</div>
    <div class="hero-title">Silent Health Risk Predictor</div>
    <p class="hero-sub">Enter your health details below to receive a personalised risk assessment and preventive recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ── Model not found warning ───────────────────────────────────
if missing_files:
    st.error(f"""
**Model files not found:** {', '.join(missing_files)}

Make sure you have run **Cell 10** in your Colab notebook to save the model files,
then place them in the same folder as this script.
""")
    st.stop()


# ── Prediction function ───────────────────────────────────────
def predict(inputs):
    bmi  = inputs['weight'] / ((inputs['height'] / 100) ** 2)
    whtr = inputs['waist'] / inputs['height']

    if bmi < 18.5:   bmi_cat = 'Under'
    elif bmi < 25:   bmi_cat = 'Normal'
    elif bmi < 30:   bmi_cat = 'Overweight'
    else:            bmi_cat = 'Obese'

    sys_bp = inputs['bp_sys']
    if sys_bp >= 140:   bp_cat = 'Stage2'
    elif sys_bp >= 130: bp_cat = 'Stage1'
    elif sys_bp >= 120: bp_cat = 'Elevated'
    else:               bp_cat = 'Normal'

    row = {
        'Age':                  inputs['age'],
        'Height_cm':            inputs['height'],
        'Weight_kg':            inputs['weight'],
        'Waist_cm':             inputs['waist'],
        'Sleep_Hours':          inputs['sleep'],
        'Water_Intake_L':       inputs['water'],
        'Screen_Time_hr':       6.0,
        'Daily_Steps':          6000,
        'Blood_Sugar_Fasting':  inputs['sugar'],
        'Cholesterol':          inputs['chol'],
        'BP_Systolic':          sys_bp,
        'BP_Diastolic':         inputs['bp_dia'],
        'BMI':                  bmi,
        'WHtR':                 whtr,
        'Activity_Level_enc':   ['Sedentary','Light','Moderate','Active','Very Active'].index(inputs['activity']),
        'Junk_Food_enc':        ['Never','Rarely','Sometimes','Often'].index(inputs['junk']),
        'Stress_Level_enc':     ['Low','Medium','High'].index(inputs['stress']),
        'Fatigue_enc':          3 if inputs['fatigue'] else 0,
        'Thirst_Urination_enc': 3 if inputs['thirst'] else 0,
        'Breath_SOB_enc':       3 if inputs['breath'] else 0,
        'BMI_Category_enc':     ['Under','Normal','Overweight','Obese'].index(bmi_cat),
        'BP_Category_enc':      ['Normal','Elevated','Stage1','Stage2'].index(bp_cat),
        'Gender_enc':           1 if inputs['gender'] == 'Male' else 0,
        'Smoking_enc':          {'No': 0, 'Former': 1, 'Yes': 2}[inputs['smoking']],
        'Alcohol_enc':          {'No': 0, 'Occasional': 1, 'Regular': 2}[inputs['alcohol']],
    }

    for prof in ['Software','Doctor','Teacher','Business','Government','Unemployed','Retired']:
        row[f'Profession_{prof}'] = 0
    for mar in ['Married','Divorced']:
        row[f'Marital_Status_{mar}'] = 0
    for fam in ['Diabetes','Hypertension','Heart Disease','Multiple']:
        row[f'Family_History_{fam}'] = 1 if fam == inputs['family_hist'] else 0
    for med in ['Diabetes','Hypertension','Asthma','Multiple']:
        row[f'Medical_Background_{med}'] = 1 if med == inputs['medical_back'] else 0

    for f in all_features:
        if f not in row:
            row[f] = 0

    df_input = pd.DataFrame([row])[all_features]

    num_cols = ['Age','Height_cm','Weight_kg','Waist_cm','Sleep_Hours',
                'Water_Intake_L','Screen_Time_hr','Daily_Steps',
                'Blood_Sugar_Fasting','Cholesterol','BP_Systolic','BP_Diastolic','BMI','WHtR']
    num_cols = [c for c in num_cols if c in df_input.columns]
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    proba     = model.predict_proba(df_input)[0]
    predicted = risk_le.inverse_transform([proba.argmax()])[0]
    classes   = list(risk_le.classes_)
    prob_dict = {c: round(p * 100, 1) for c, p in zip(classes, proba)}

    return predicted, prob_dict, bmi, whtr, bmi_cat, bp_cat


# ── Layout: 3 columns ─────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 1, 1], gap="large")

# ════════════════════════════════
# LEFT COLUMN — Personal + Body
# ════════════════════════════════
with col_left:
    st.markdown('<div class="section-header">Personal Info</div>', unsafe_allow_html=True)
    age    = st.slider("Age", 18, 90, 30)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=25, max_value=250, value=70)
    waist  = st.number_input("Waist circumference (cm)", min_value=40, max_value=200, value=80)

    st.markdown('<div class="section-header">Lifestyle</div>', unsafe_allow_html=True)
    sleep    = st.slider("Sleep hours per night", 4, 12, 7)
    water    = st.slider("Water intake (L/day)", 0.5, 5.0, 2.0, step=0.1)
    activity = st.selectbox("Activity level",
                            ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                            index=2)
    junk     = st.selectbox("Junk food frequency",
                            ["Never", "Rarely", "Sometimes", "Often"], index=1)


# ════════════════════════════════
# MIDDLE COLUMN — Health + History
# ════════════════════════════════
with col_mid:
    st.markdown('<div class="section-header">Health Measurements</div>', unsafe_allow_html=True)
    smoking  = st.radio("Smoking", ["No", "Former", "Yes"], horizontal=True)
    alcohol  = st.radio("Alcohol consumption", ["No", "Occasional", "Regular"], horizontal=True)

    bp_str   = st.text_input("Blood pressure (systolic/diastolic)", value="120/80")
    try:
        parts  = bp_str.strip().split('/')
        bp_sys = int(parts[0])
        bp_dia = int(parts[1]) if len(parts) > 1 else 80
    except Exception:
        bp_sys, bp_dia = 120, 80
        st.warning("Enter BP as 120/80 format")

    sugar    = st.number_input("Fasting blood sugar (mg/dL)", min_value=50, max_value=400, value=90)
    chol     = st.number_input("Total cholesterol (mg/dL)",   min_value=100, max_value=400, value=190)

    st.markdown('<div class="section-header">History</div>', unsafe_allow_html=True)
    stress       = st.selectbox("Stress level", ["Low", "Medium", "High"])
    family_hist  = st.selectbox("Family medical history",
                                ["None", "Diabetes", "Hypertension", "Heart Disease", "Multiple"])
    medical_back = st.selectbox("Your past / present conditions",
                                ["None", "Diabetes", "Hypertension", "Asthma", "Multiple"])

    st.markdown('<div class="section-header">Symptoms (tick if often)</div>', unsafe_allow_html=True)
    fatigue = st.checkbox("Extreme fatigue / persistent tiredness")
    thirst  = st.checkbox("Excessive thirst / frequent urination")
    breath  = st.checkbox("Shortness of breath while walking")


# ════════════════════════════════
# RIGHT COLUMN — Result
# ════════════════════════════════
with col_right:
    st.markdown('<div class="section-header">Your Assessment</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict My Risk")

    if predict_btn:
        inputs = dict(
            age=age, gender=gender, height=height, weight=weight, waist=waist,
            sleep=sleep, water=water, activity=activity, junk=junk,
            smoking=smoking, alcohol=alcohol, bp_sys=bp_sys, bp_dia=bp_dia,
            sugar=sugar, chol=chol, stress=stress,
            family_hist=family_hist, medical_back=medical_back,
            fatigue=fatigue, thirst=thirst, breath=breath,
        )

        with st.spinner("Analysing your health profile…"):
            predicted, prob_dict, bmi, whtr, bmi_cat, bp_cat = predict(inputs)

        # ── Risk result card ──
        color_map   = {'Low': 'low', 'Moderate': 'mod', 'High': 'high'}
        icon_map    = {'Low': '✦', 'Moderate': '◈', 'High': '⬟'}
        css_map     = {'Low': 'result-low', 'Moderate': 'result-moderate', 'High': 'result-high'}
        risk_class  = color_map[predicted]
        result_css  = css_map[predicted]

        st.markdown(f"""
        <div class="{result_css}">
            <div class="risk-label" style="color:#718096">Predicted Risk Level</div>
            <div class="risk-value risk-{risk_class}">{icon_map[predicted]} {predicted}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown("<br>", unsafe_allow_html=True)
        low_p  = prob_dict.get('Low', 0)
        mod_p  = prob_dict.get('Moderate', 0)
        high_p = prob_dict.get('High', 0)

        st.markdown(f"""
        <div style="background:#1a1f2e; border:1px solid #2a3142; border-radius:12px; padding:1.2rem;">
            <div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#4a5568; margin-bottom:1rem;">Model Confidence</div>
            <div class="prob-row">
                <div class="prob-label" style="color:#68d391">Low</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill-low" style="width:{low_p}%"></div></div>
                <div class="prob-pct" style="color:#68d391">{low_p}%</div>
            </div>
            <div class="prob-row">
                <div class="prob-label" style="color:#f6ad55">Moderate</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill-mod" style="width:{mod_p}%"></div></div>
                <div class="prob-pct" style="color:#f6ad55">{mod_p}%</div>
            </div>
            <div class="prob-row">
                <div class="prob-label" style="color:#fc8181">High</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill-high" style="width:{high_p}%"></div></div>
                <div class="prob-pct" style="color:#fc8181">{high_p}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Body metrics ──
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-pill">
                <div class="metric-pill-label">BMI</div>
                <div class="metric-pill-value">{bmi:.1f} <span style="font-size:0.7rem;color:#4a5568;font-weight:400">{bmi_cat}</span></div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">WHtR</div>
                <div class="metric-pill-value">{whtr:.2f} <span style="font-size:0.7rem;color:#4a5568;font-weight:400">{'⚠ High' if whtr > 0.5 else 'OK'}</span></div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">BP Category</div>
                <div class="metric-pill-value" style="font-size:0.95rem">{bp_cat}</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Blood Sugar</div>
                <div class="metric-pill-value">{sugar} <span style="font-size:0.7rem;color:#4a5568;font-weight:400">mg/dL</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Suggestion ──
        suggestions = {
            'Low':      "Your health indicators look good. Keep up the healthy habits, stay active, and schedule an annual checkup to stay on track.",
            'Moderate': "Several risk factors need attention. Increase physical activity to at least 30 min/day, cut down on junk food, and monitor your blood pressure and sugar every 3 months.",
            'High':     "Multiple high-risk indicators detected. Please consult a physician within the next 2 weeks, follow a strict low-sugar low-fat diet, aim for 8000+ steps daily, and act on any smoking or alcohol habits immediately."
        }
        icons = {'Low': '💚', 'Moderate': '🟡', 'High': '🔴'}

        st.markdown(f"""
        <div class="suggestion-box" style="margin-top:1rem;">
            <div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#4a5568; margin-bottom:0.5rem;">{icons[predicted]} Preventive Suggestion</div>
            <p>{suggestions[predicted]}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="background:#1a1f2e; border:1px dashed #2a3142; border-radius:12px;
                    padding:3rem 2rem; text-align:center; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.3">◎</div>
            <div style="color:#4a5568; font-size:0.9rem; line-height:1.6">
                Fill in your details on the left<br>and click <strong style="color:#63b3ed">Predict My Risk</strong><br>to see your assessment
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#2d3748; font-size:0.75rem; padding:1rem 0; border-top:1px solid #1e2433;">
    For educational purposes only — not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)

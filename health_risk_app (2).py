
!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Silent Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f1117; }
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 50%, #1a1f2e 100%);
    border: 1px solid #2a3142; border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem;
}
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: #e2e8f0; margin: 0 0 0.5rem 0; }
.hero-sub   { font-size: 1rem; color: #718096; margin: 0; font-weight: 300; }
.hero-badge {
    display: inline-block; background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.3); color: #63b3ed;
    font-size: 0.75rem; font-weight: 500; padding: 4px 12px;
    border-radius: 20px; margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase;
}
.section-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a5568; margin: 1.8rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e2433;
}
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-size: 1rem !important;
    font-weight: 600 !important; width: 100% !important; margin-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    required = ['best_risk_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'feature_names.pkl']
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    model   = joblib.load('best_risk_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    risk_le = joblib.load('label_encoder.pkl')
    features= joblib.load('feature_names.pkl')
    return model, scaler, risk_le, features, []

model, scaler, risk_le, all_features, missing_files = load_models()

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Health Assessment</div>
    <div class="hero-title">Silent Health Risk Predictor</div>
    <p class="hero-sub">Enter your health details to receive a personalised risk assessment.</p>
</div>
""", unsafe_allow_html=True)

if missing_files:
    st.error(f"Model files not found: {missing_files}. Upload pkl files to your repo.")
    st.stop()

def predict(inputs):
    bmi  = inputs['weight'] / ((inputs['height'] / 100) ** 2)
    whtr = inputs['waist']  / inputs['height']
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
        'Age': inputs['age'], 'Height_cm': inputs['height'], 'Weight_kg': inputs['weight'],
        'Waist_cm': inputs['waist'], 'Sleep_Hours': inputs['sleep'],
        'Water_Intake_L': inputs['water'], 'Screen_Time_hr': 6.0, 'Daily_Steps': 6000,
        'Blood_Sugar_Fasting': inputs['sugar'], 'Cholesterol': inputs['chol'],
        'BP_Systolic': sys_bp, 'BP_Diastolic': inputs['bp_dia'], 'BMI': bmi, 'WHtR': whtr,
        'Activity_Level_enc':   ['Sedentary','Light','Moderate','Active','Very Active'].index(inputs['activity']),
        'Junk_Food_enc':        ['Never','Rarely','Sometimes','Often'].index(inputs['junk']),
        'Stress_Level_enc':     ['Low','Medium','High'].index(inputs['stress']),
        'Fatigue_enc':          3 if inputs['fatigue'] else 0,
        'Thirst_Urination_enc': 3 if inputs['thirst'] else 0,
        'Breath_SOB_enc':       3 if inputs['breath'] else 0,
        'BMI_Category_enc':     ['Under','Normal','Overweight','Obese'].index(bmi_cat),
        'BP_Category_enc':      ['Normal','Elevated','Stage1','Stage2'].index(bp_cat),
        'Gender_enc':           1 if inputs['gender'] == 'Male' else 0,
        'Smoking_enc':          {'No':0,'Former':1,'Yes':2}[inputs['smoking']],
        'Alcohol_enc':          {'No':0,'Occasional':1,'Regular':2}[inputs['alcohol']],
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
        if f not in row: row[f] = 0
    df_in = pd.DataFrame([row])[all_features]
    num_cols = [c for c in ['Age','Height_cm','Weight_kg','Waist_cm','Sleep_Hours',
                'Water_Intake_L','Screen_Time_hr','Daily_Steps','Blood_Sugar_Fasting',
                'Cholesterol','BP_Systolic','BP_Diastolic','BMI','WHtR'] if c in df_in.columns]
    df_in[num_cols] = scaler.transform(df_in[num_cols])
    proba     = model.predict_proba(df_in)[0]
    predicted = risk_le.inverse_transform([proba.argmax()])[0]
    classes   = list(risk_le.classes_)
    prob_dict = {c: round(p*100,1) for c,p in zip(classes,proba)}
    return predicted, prob_dict, bmi, whtr, bmi_cat, bp_cat

col_left, col_mid, col_right = st.columns([1,1,1], gap="large")

with col_left:
    st.markdown('<div class="section-header">Personal Info</div>', unsafe_allow_html=True)
    age    = st.slider("Age", 18, 90, 30)
    gender = st.radio("Gender", ["Male","Female"], horizontal=True)
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)",  25, 250,  70)
    waist  = st.number_input("Waist circumference (cm)", 40, 200, 80)
    st.markdown('<div class="section-header">Lifestyle</div>', unsafe_allow_html=True)
    sleep    = st.slider("Sleep hours per night", 4, 12, 7)
    water    = st.slider("Water intake (L/day)", 0.5, 5.0, 2.0, step=0.1)
    activity = st.selectbox("Activity level", ["Sedentary","Light","Moderate","Active","Very Active"], index=2)
    junk     = st.selectbox("Junk food frequency", ["Never","Rarely","Sometimes","Often"], index=1)

with col_mid:
    st.markdown('<div class="section-header">Health Measurements</div>', unsafe_allow_html=True)
    smoking  = st.radio("Smoking", ["No","Former","Yes"], horizontal=True)
    alcohol  = st.radio("Alcohol", ["No","Occasional","Regular"], horizontal=True)
    bp_str   = st.text_input("Blood pressure (systolic/diastolic)", value="120/80")
    try:
        parts = bp_str.strip().split('/')
        bp_sys = int(parts[0]); bp_dia = int(parts[1]) if len(parts)>1 else 80
    except: bp_sys, bp_dia = 120, 80
    sugar = st.number_input("Fasting blood sugar (mg/dL)", 50, 400, 90)
    chol  = st.number_input("Total cholesterol (mg/dL)",  100, 400, 190)
    st.markdown('<div class="section-header">History</div>', unsafe_allow_html=True)
    stress       = st.selectbox("Stress level", ["Low","Medium","High"])
    family_hist  = st.selectbox("Family medical history", ["None","Diabetes","Hypertension","Heart Disease","Multiple"])
    medical_back = st.selectbox("Past/present conditions", ["None","Diabetes","Hypertension","Asthma","Multiple"])
    st.markdown('<div class="section-header">Symptoms (tick if often)</div>', unsafe_allow_html=True)
    fatigue = st.checkbox("Extreme fatigue / persistent tiredness")
    thirst  = st.checkbox("Excessive thirst / frequent urination")
    breath  = st.checkbox("Shortness of breath while walking")

with col_right:
    st.markdown('<div class="section-header">Your Assessment</div>', unsafe_allow_html=True)
    if st.button("🔍 Predict My Risk"):
        inputs = dict(age=age, gender=gender, height=height, weight=weight, waist=waist,
                      sleep=sleep, water=water, activity=activity, junk=junk,
                      smoking=smoking, alcohol=alcohol, bp_sys=bp_sys, bp_dia=bp_dia,
                      sugar=sugar, chol=chol, stress=stress, family_hist=family_hist,
                      medical_back=medical_back, fatigue=fatigue, thirst=thirst, breath=breath)
        with st.spinner("Analysing…"):
            predicted, prob_dict, bmi, whtr, bmi_cat, bp_cat = predict(inputs)
        color = {"Low":"🟢","Moderate":"🟡","High":"🔴"}
        st.success(f"{color[predicted]}  Predicted Risk Level: **{predicted}**")
        low_p  = prob_dict.get('Low', 0)
        mod_p  = prob_dict.get('Moderate', 0)
        high_p = prob_dict.get('High', 0)
        st.markdown("**Model Confidence**")
        st.progress(int(low_p),  text=f"🟢 Low: {low_p}%")
        st.progress(int(mod_p),  text=f"🟡 Moderate: {mod_p}%")
        st.progress(int(high_p), text=f"🔴 High: {high_p}%")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BMI", f"{bmi:.1f}", bmi_cat)
        col2.metric("WHtR", f"{whtr:.2f}", "⚠ High" if whtr>0.5 else "OK")
        col3.metric("BP", bp_cat)
        col4.metric("Sugar", f"{sugar} mg/dL")
        suggestions = {
            'Low':      "Maintain healthy habits. Annual checkup recommended.",
            'Moderate': "Increase activity, reduce junk food, monitor BP/sugar every 3 months.",
            'High':     "Consult a physician soon. Make lifestyle changes immediately."
        }
        st.info(f"💡 **Suggestion:** {suggestions[predicted]}")
    else:
        st.info("Fill in your details and click **Predict My Risk**")

st.markdown("---")
st.caption("For educational purposes only — not a substitute for professional medical advice.")

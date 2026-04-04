import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Silent Health Risk Predictor", page_icon="🏥", layout="wide")

@st.cache_resource
def load_models():
    required = ['best_risk_model.pkl','scaler.pkl','label_encoder.pkl','feature_names.pkl']
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing
    return joblib.load('best_risk_model.pkl'), joblib.load('scaler.pkl'),            joblib.load('label_encoder.pkl'), joblib.load('feature_names.pkl'), []

model, scaler, risk_le, all_features, missing_files = load_models()

st.title("🏥 Silent Health Risk Predictor")

if missing_files:
    st.error(f"Missing model files: {missing_files}")
    st.stop()

def predict(inputs):
    bmi  = inputs['weight'] / ((inputs['height'] / 100) ** 2)
    whtr = inputs['waist']  / inputs['height']
    bmi_cat = 'Under' if bmi<18.5 else 'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'
    sys_bp  = inputs['bp_sys']
    bp_cat  = 'Normal' if sys_bp<120 else 'Elevated' if sys_bp<130 else 'Stage1' if sys_bp<140 else 'Stage2'
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
        'Gender_enc':           1 if inputs['gender']=='Male' else 0,
        'Smoking_enc':          {'No':0,'Former':1,'Yes':2}[inputs['smoking']],
        'Alcohol_enc':          {'No':0,'Occasional':1,'Regular':2}[inputs['alcohol']],
    }
    for p in ['Software','Doctor','Teacher','Business','Government','Unemployed','Retired']:
        row[f'Profession_{p}'] = 0
    for m in ['Married','Divorced']:
        row[f'Marital_Status_{m}'] = 0
    for f in ['Diabetes','Hypertension','Heart Disease','Multiple']:
        row[f'Family_History_{f}'] = 1 if f==inputs['family_hist'] else 0
    for f in ['Diabetes','Hypertension','Asthma','Multiple']:
        row[f'Medical_Background_{f}'] = 1 if f==inputs['medical_back'] else 0
    for f in all_features:
        if f not in row: row[f] = 0
    df_in = pd.DataFrame([row])[all_features]
    num_cols = [c for c in ['Age','Height_cm','Weight_kg','Waist_cm','Sleep_Hours',
                'Water_Intake_L','Screen_Time_hr','Daily_Steps','Blood_Sugar_Fasting',
                'Cholesterol','BP_Systolic','BP_Diastolic','BMI','WHtR'] if c in df_in.columns]
    df_in[num_cols] = scaler.transform(df_in[num_cols])
    proba     = model.predict_proba(df_in)[0]
    predicted = risk_le.inverse_transform([proba.argmax()])[0]
    prob_dict = {c: round(p*100,1) for c,p in zip(risk_le.classes_, proba)}
    return predicted, prob_dict, bmi, whtr, bmi_cat, bp_cat

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age    = st.slider("Age", 18, 90, 30)
    gender = st.radio("Gender", ["Male","Female"], horizontal=True)
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)",  25, 250,  70)
    waist  = st.number_input("Waist (cm)",   40, 200,  80)
    st.subheader("Lifestyle")
    sleep    = st.slider("Sleep hours/night", 4, 12, 7)
    water    = st.slider("Water intake L/day", 0.5, 5.0, 2.0, step=0.1)
    activity = st.selectbox("Activity level", ["Sedentary","Light","Moderate","Active","Very Active"], index=2)
    junk     = st.selectbox("Junk food", ["Never","Rarely","Sometimes","Often"], index=1)

with col2:
    st.subheader("Health")
    smoking = st.radio("Smoking",  ["No","Former","Yes"], horizontal=True)
    alcohol = st.radio("Alcohol",  ["No","Occasional","Regular"], horizontal=True)
    bp_str  = st.text_input("Blood Pressure (e.g. 120/80)", "120/80")
    try:
        parts  = bp_str.split('/')
        bp_sys = int(parts[0]); bp_dia = int(parts[1]) if len(parts)>1 else 80
    except:
        bp_sys, bp_dia = 120, 80
    sugar = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 400, 90)
    chol  = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)
    st.subheader("History")
    stress       = st.selectbox("Stress level", ["Low","Medium","High"])
    family_hist  = st.selectbox("Family history", ["None","Diabetes","Hypertension","Heart Disease","Multiple"])
    medical_back = st.selectbox("Your conditions", ["None","Diabetes","Hypertension","Asthma","Multiple"])
    st.subheader("Symptoms")
    fatigue = st.checkbox("Frequent extreme fatigue")
    thirst  = st.checkbox("Excessive thirst / urination")
    breath  = st.checkbox("Shortness of breath")

with col3:
    st.subheader("Your Result")
    if st.button("🔍 Predict Risk", use_container_width=True):
        inp = dict(age=age, gender=gender, height=height, weight=weight, waist=waist,
                   sleep=sleep, water=water, activity=activity, junk=junk,
                   smoking=smoking, alcohol=alcohol, bp_sys=bp_sys, bp_dia=bp_dia,
                   sugar=sugar, chol=chol, stress=stress, family_hist=family_hist,
                   medical_back=medical_back, fatigue=fatigue, thirst=thirst, breath=breath)
        with st.spinner("Analysing..."):
            predicted, prob_dict, bmi, whtr, bmi_cat, bp_cat = predict(inp)
        icon = {"Low":"🟢","Moderate":"🟡","High":"🔴"}[predicted]
        color = {"Low":"green","Moderate":"orange","High":"red"}[predicted]
        st.markdown(f"### {icon} Risk Level: :{color}[**{predicted}**]")
        st.progress(int(prob_dict.get('Low',0)),     text=f"🟢 Low: {prob_dict.get('Low',0)}%")
        st.progress(int(prob_dict.get('Moderate',0)),text=f"🟡 Moderate: {prob_dict.get('Moderate',0)}%")
        st.progress(int(prob_dict.get('High',0)),    text=f"🔴 High: {prob_dict.get('High',0)}%")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("BMI", f"{bmi:.1f}", bmi_cat)
        c2.metric("WHtR", f"{whtr:.2f}", "⚠ High" if whtr>0.5 else "OK")
        c3.metric("BP", bp_cat)
        c4.metric("Sugar", f"{sugar}")
        tips = {
            'Low':      "✅ Maintain healthy habits. Annual checkup recommended.",
            'Moderate': "⚠️ Increase activity, reduce junk food, monitor BP/sugar every 3 months.",
            'High':     "🚨 Consult a physician soon. Immediate lifestyle changes needed."
        }
        st.info(tips[predicted])
    else:
        st.info("Fill in your details and click Predict Risk")

st.caption("For educational purposes only — not a substitute for professional medical advice.")

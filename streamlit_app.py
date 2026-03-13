# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and tools
@st.cache_resource
def load_model():
    model = joblib.load('best_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, label_encoder, feature_names

model, scaler, label_encoder, feature_names = load_model()

# Title
st.title("🏥 Intelligent Silent Health Risk Predictor")
st.markdown("""
> Detect silent health risks early. For educational purposes only.
""")

# Sidebar inputs
with st.sidebar:
    st.header("👤 Your Information")

    age = st.slider("Age", 18, 90, 35)
    gender = st.radio("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 100, 220, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    waist = st.number_input("Waist Size (cm)", 50, 150, 85)

    sleep = st.slider("Sleep Hours", 4, 12, 7)
    water = st.slider("Water Intake (L/day)", 0.5, 5.0, 2.0)
    activity = st.selectbox("Activity Level", [
        "Sedentary", "Light", "Moderate", "Active", "Very Active"
    ])
    junk = st.selectbox("Junk Food Consumption", [
        "Never", "Rarely", "Sometimes", "Often"
    ])

    smoking = st.radio("Smoking", ["No", "Yes", "Former"])
    alcohol = st.radio("Alcohol", ["No", "Occasional", "Regular"])

    bp_input = st.text_input("Blood Pressure (e.g., 120/80)", "120/80")
    sugar = st.number_input("Fasting Blood Sugar (mg/dL)", 60, 300, 90)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)

    stress = st.select_slider("Stress Level", ["Low", "Medium", "High"])
    family_hist = st.selectbox("Family Medical History", [
        "None", "Diabetes", "Hypertension", "Heart Disease", "Multiple"
    ])
    medical_back = st.selectbox("Past/Present Diseases", [
        "None", "Diabetes", "Hypertension", "Asthma", "Multiple"
    ])

    fatigue = st.checkbox("Extreme fatigue / tiredness")
    thirst = st.checkbox("Excessive thirst or frequent urination")
    breath = st.checkbox("Shortness of breath while walking")

# Process BP input
try:
    systolic_bp = int(bp_input.split('/')[0])
except:
    systolic_bp = 120
diastolic_bp = 80  # placeholder

# Compute BMI and WHtR
bmi = weight / ((height / 100) ** 2)
whtr = waist / height

# Categorize BMI
if bmi < 18.5:
    bmi_cat = 'Under'
elif bmi < 25:
    bmi_cat = 'Normal'
elif bmi < 30:
    bmi_cat = 'Overweight'
else:
    bmi_cat = 'Obese'

# Categorize BP
if systolic_bp >= 140:
    bp_cat = 'Stage2'
elif systolic_bp >= 130:
    bp_cat = 'Stage1'
elif systolic_bp >= 120:
    bp_cat = 'Elevated'
else:
    bp_cat = 'Normal'

# Encode categorical variables
input_data = {
    'Age': age,
    'Height_cm': height,
    'Weight_kg': weight,
    'Waist_cm': waist,
    'Sleep_Hours': sleep,
    'Water_Intake_L': water,
    'Screen_Time_hr': 6,  # default
    'Daily_Steps': 5000,   # default
    'Blood_Sugar_Fasting': sugar,
    'Cholesterol': chol,
    'BP_Systolic': systolic_bp,
    'BP_Diastolic': diastolic_bp,
    'BMI': bmi,
    'WHtR': whtr,

    # Ordinal encoded
    'Activity_Level_enc': ["Sedentary", "Light", "Moderate", "Active", "Very Active"].index(activity),
    'Junk_Food_enc': ["Never", "Rarely", "Sometimes", "Often"].index(junk),
    'Stress_Level_enc': ["Low", "Medium", "High"].index(stress),
    'Fatigue_enc': 3 if fatigue else 0,
    'Thirst_Urination_enc': 3 if thirst else 0,
    'Breath_SOB_enc': 3 if breath else 0,
    'BMI_Category_enc': ["Under", "Normal", "Overweight", "Obese"].index(bmi_cat),
    'BP_Category_enc': ["Normal", "Elevated", "Stage1", "Stage2"].index(bp_cat),

    # Binary
    'Gender_enc': 1 if gender == "Male" else 0,
    'Smoking_enc': 1 if smoking == "Yes" else 0,
    'Alcohol_enc': 0 if alcohol == "No" else (1 if alcohol == "Occasional" else 2),
}

# One-Hot Encoding (match training data)
for prof in ['Software', 'Doctor', 'Teacher', 'Business', 'Government', 'Unemployed', 'Retired']:
    input_data[f'Profession_{prof}'] = 1 if prof == 'Software' else 0  # dummy

for marital in ['Married', 'Divorced']:
    input_data[f'Marital_Status_{marital}'] = 1 if marital == 'Single' else 0

for fam in ['Diabetes', 'Hypertension', 'Heart Disease', 'Multiple']:
    input_data[f'Family_History_{fam}'] = 1 if fam == family_hist else 0

for med in ['Diabetes', 'Hypertension', 'Asthma', 'Multiple']:
    input_data[f'Medical_Background_{med}'] = 1 if med == medical_back else 0

# Ensure all features exist
for f in feature_names:
    if f not in input_data:
        input_data[f] = 0

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[feature_names]

# Scale numerical features
num_cols = ['Age', 'Height_cm', 'Weight_kg', 'Waist_cm', 'Sleep_Hours',
            'Water_Intake_L', 'Screen_Time_hr', 'Daily_Steps',
            'Blood_Sugar_Fasting', 'Cholesterol', 'BP_Systolic', 'BP_Diastolic',
            'BMI', 'WHtR']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict button
if st.button("🔍 Predict Risk"):
    # Predict probabilities
    proba = model.predict_proba(input_df)[0]
    risk_levels = label_encoder.classes_  # ['Low', 'Moderate', 'High']

    # Display results
    st.subheader("📊 Risk Assessment Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("Risk Probability Distribution:")
        for level, prob in zip(risk_levels, proba):
            bar = '█' * int(prob * 30)
            st.write(f"{level:<10} → {prob*100:5.1f}%  {bar}")

    with col2:
        predicted_risk = risk_levels[proba.argmax()]
        st.metric("Predicted Risk", predicted_risk, f"{proba.max()*100:.1f}%")

    # Preventive suggestions
    st.markdown("💡 **Preventive Suggestions:**")
    if predicted_risk == "Low":
        st.success("✅ Maintain healthy habits. Annual check-up recommended.")
    elif predicted_risk == "Moderate":
        st.warning("⚠️ Increase activity, reduce junk food, monitor BP/sugar every 3 months.")
    else:
        st.error("🚨 Consult a physician soon. Consider lifestyle change and medication.")

    # Show stats
    st.markdown("---")
    st.caption(f"BMI: {bmi:.1f} ({bmi_cat}) | WHtR: {whtr:.2f}")
    st.caption(f"Systolic BP: {systolic_bp} mmHg | Fasting Sugar: {sugar} mg/dL")

# Footer
st.markdown("---")
st.markdown("""
> ⚠️ **Disclaimer**: This tool is for educational and research purposes only.  
> It does **not** replace professional medical advice, diagnosis, or treatment.
""")
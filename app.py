import streamlit as st
import numpy as np
import pickle

# Load the trained models
with open('stroke_risk_classification_model.pkl', 'rb') as f:
    model_classification = pickle.load(f)

with open('stroke_risk_regression_model.pkl', 'rb') as f:
    model_regression = pickle.load(f)

# Streamlit UI
st.title("🧠 Stroke Risk Prediction App")

# User inputs
age = st.slider("Age", 0, 100, 30)
gender_input = st.selectbox("Gender", ['Male', 'Female'])
gender_encoded = 1 if gender_input == 'Male' else 0

# Symptom inputs (14 checkboxes)
chest_pain = st.checkbox("Chest Pain")
high_blood_pressure = st.checkbox("High Blood Pressure")
irregular_heartbeat = st.checkbox("Irregular Heartbeat")
shortness_of_breath = st.checkbox("Shortness of Breath")
fatigue_weakness = st.checkbox("Fatigue or Weakness")
dizziness = st.checkbox("Dizziness")
swelling_edema = st.checkbox("Swelling or Edema")
neck_jaw_pain = st.checkbox("Neck or Jaw Pain")
excessive_sweating = st.checkbox("Excessive Sweating")
persistent_cough = st.checkbox("Persistent Cough")
nausea_vomiting = st.checkbox("Nausea or Vomiting")
chest_discomfort = st.checkbox("Chest Discomfort")
cold_hands_feet = st.checkbox("Cold Hands or Feet")
snoring_sleep_apnea = st.checkbox("Snoring or Sleep Apnea")
anxiety_doom = st.checkbox("Anxiety or Sense of Doom")  # 14th symptom

# Prediction
if st.button("Predict Stroke Risk"):
    # Build input array (17 features)
    input_data = np.array([[ 
        age,  # Age (numeric value)
        gender_encoded,  # Gender (0 or 1)
        int(chest_pain),  # Chest Pain (binary)
        int(high_blood_pressure),  # High Blood Pressure (binary)
        int(irregular_heartbeat),  # Irregular Heartbeat (binary)
        int(shortness_of_breath),  # Shortness of Breath (binary)
        int(fatigue_weakness),  # Fatigue or Weakness (binary)
        int(dizziness),  # Dizziness (binary)
        int(swelling_edema),  # Swelling or Edema (binary)
        int(neck_jaw_pain),  # Neck or Jaw Pain (binary)
        int(excessive_sweating),  # Excessive Sweating (binary)
        int(persistent_cough),  # Persistent Cough (binary)
        int(nausea_vomiting),  # Nausea or Vomiting (binary)
        int(chest_discomfort),  # Chest Discomfort (binary)
        int(cold_hands_feet),  # Cold Hands or Feet (binary)
        int(snoring_sleep_apnea),  # Snoring or Sleep Apnea (binary)
        int(anxiety_doom),  # Anxiety or Sense of Doom (binary)
    ]])

    # Debugging
    st.write(f"Input data shape: {input_data.shape}")

    # Make predictions
    class_pred = model_classification.predict(input_data)[0]
    reg_pred = model_regression.predict(input_data)[0]

    # Show results
    st.success(f"Risk Category: {'At Risk' if class_pred == 1 else 'Not At Risk'}")
    st.info(f"Estimated Stroke Risk Percentage: {reg_pred:.2f}%")

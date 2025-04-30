import streamlit as st
import numpy as np
import joblib

# ------------------ Load Models ------------------ #
@st.cache_resource
def load_models():
    try:
        reg = joblib.load('stroke_risk_regression_model.pkl')
        clf = joblib.load('stroke_risk_classification_model.pkl')
        return reg, clf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

reg_model, clf_model = load_models()

# ------------------ Streamlit UI ------------------ #
st.title("🧠 Stroke Risk Prediction App")
st.markdown("Estimate a patient's **stroke risk percentage** and classify whether they are **at risk** based on symptoms.")

# Patient Inputs
st.header("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 0, 100, 30)
with col2:
    st.subheader("Select Gender")
    male = st.checkbox("Male")
    female = st.checkbox("Female")

# Gender Encoding
if male and female:
    st.warning("⚠️Please select only one gender.")
    gender_encoded = None
elif not male and not female:
    gender_encoded = None
else:
    gender_encoded = 1 if male else 0

# Symptom Checkboxes
st.subheader("Symptoms Checklist")
symptoms = {
    "Chest Pain": st.checkbox("Chest Pain"),
    "High Blood Pressure": st.checkbox("High Blood Pressure"),
    "Irregular Heartbeat": st.checkbox("Irregular Heartbeat"),
    "Shortness of Breath": st.checkbox("Shortness of Breath"),
    "Fatigue or Weakness": st.checkbox("Fatigue or Weakness"),
    "Dizziness": st.checkbox("Dizziness"),
    "Swelling or Edema": st.checkbox("Swelling or Edema"),
    "Neck or Jaw Pain": st.checkbox("Neck or Jaw Pain"),
    "Excessive Sweating": st.checkbox("Excessive Sweating"),
    "Persistent Cough": st.checkbox("Persistent Cough"),
    "Nausea or Vomiting": st.checkbox("Nausea or Vomiting"),
    "Chest Discomfort": st.checkbox("Chest Discomfort"),
    "Cold Hands or Feet": st.checkbox("Cold Hands or Feet"),
    "Snoring or Sleep Apnea": st.checkbox("Snoring or Sleep Apnea"),
    "Anxiety or Sense of Doom": st.checkbox("Anxiety or Sense of Doom"),
}

# ------------------ Prediction ------------------ #
if st.button("Predict Stroke Risk"):
    if gender_encoded is None:
        st.error("Please select exactly one gender.")
    elif not reg_model or not clf_model:
        st.error("Model files not loaded correctly.")
    else:
        # Input feature vector (17 features)
        feature_vector = np.array([[
            age,
            gender_encoded,
            *[int(value) for value in symptoms.values()]
        ]])

        # Predict stroke risk percentage (regression)
        stroke_risk = reg_model.predict(feature_vector)[0]

        # Add predicted percentage to features (18 total)
        classification_input = np.hstack([feature_vector, [[stroke_risk]]])

        # Predict class (0 = Not At Risk, 1 = At Risk)
        risk_class = clf_model.predict(classification_input)[0]

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        st.success(f"🩺 Risk Category: **{'At Risk' if risk_class == 1 else 'Not At Risk'}**")
        st.info(f"📊 Estimated Stroke Risk: **{stroke_risk:.2f}%**")

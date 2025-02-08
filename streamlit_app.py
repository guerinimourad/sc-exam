
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("Cancer Recurrence Prediction")

# Input fields for user
age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
race = st.selectbox("Race/Ethnicity", [0, 1, 2, 3, 4])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", [0, 1, 2], format_func=lambda x: ["Non-smoker", "Smoker", "Former smoker"][x])
tumor_size = st.number_input("Tumor Size (cm)", min_value=0.1, max_value=50.0, value=5.0)

# Predict button
if st.button("Predict Recurrence"):
    input_data = np.array([[age, gender, race, bmi, smoking_status, tumor_size]])
    prediction = model.predict(input_data)
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Predicted Recurrence: **{result}**")

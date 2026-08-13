import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

model = joblib.load("best_diabetes_model.pkl")

st.title("🩺 Diabetes Prediction")
st.write("Enter the patient's health information below.")

st.divider()

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=40
    )

    bmi = st.number_input(
        "BMI",
        min_value=17.0,
        max_value=42.0,
        value=25.0,
        step=0.1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=70.0,
        max_value=220.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=90,
        max_value=999,
        value=120
    )

with col2:
    cholesterol = st.number_input(
        "Cholesterol",
        min_value=120,
        max_value=320,
        value=200
    )

    insulin = st.number_input(
        "Insulin",
        min_value=15,
        max_value=300,
        value=100
    )

    hba1c = st.number_input(
        "HbA1c",
        min_value=4.5,
        max_value=11.5,
        value=6.5,
        step=0.1
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

st.subheader("🏃 Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:
    smoking = st.selectbox(
        "Smoking Status",
        ["Never", "Former", "Current"]
    )

with col2:
    activity = st.selectbox(
        "Physical Activity",
        ["High", "Low", "Moderate"]
    )

with col3:
    family_history = st.selectbox(
        "Family History",
        ["No", "Yes"]
    )

st.divider()

if st.button("🔍 Predict Diabetes", use_container_width=True):
    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Glucose": [glucose],
        "Blood_Pressure": [blood_pressure],
        "Cholesterol": [cholesterol],
        "Insulin": [insulin],
        "HbA1c": [hba1c],
        "Gender_Male": [1 if gender == "Male" else 0],

        "Smoking_Status_Former": [
            1 if smoking == "Former" else 0
        ],

        "Smoking_Status_Never": [
            1 if smoking == "Never" else 0
        ],

        "Physical_Activity_Low": [
            1 if activity == "Low" else 0
        ],

        "Physical_Activity_Moderate": [
            1 if activity == "Moderate" else 0
        ],

        "Family_History_Yes": [
            1 if family_history == "Yes" else 0
        ]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Diabetes Prediction: YES")
        st.warning(
            "The model predicts that the patient may have diabetes."
        )

    else:
        st.success("✅ Diabetes Prediction: NO")
        st.info(
            "The model predicts that the patient may not have diabetes."
        )

st.divider()

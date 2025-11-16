import streamlit as st

st.title("Gallbladder Necrosis Prediction App")
import joblib
import pandas as pd

# Load the trained model
try:
    # Correcting the model file name to mo_hinh_xgboost_du_doan_hoai_tu.pkl
    model = joblib.load('mo_hinh_xgboost_du_doan_hoai_tu.pkl')
    st.success("Model 'mo_hinh_xgboost_du_doan_hoai_tu.pkl' loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'mo_hinh_xgboost_du_doan_hoai_tu.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Define feature lists based on the original training script
numeric_features = [
    'age', 'bmi', 'onset_hours',
    'heart_rate', 'systolic_bp', 'diastolic_bp',
    'wbc', 'neutrophil_pct', 'nlr',
    'crp', 'ast', 'alt', 'bilirubin_total', 'creatinine'
]
categorical_features = [
    'sex', 'dm', 'hta', 'heart_disease', 'chronic_kidney',
    'fever', 'murphy_clinical',
    'wall_thickened', 'pericholecystic_fluid', 'impacted_stone',
    'gallbladder_distended', 'gas_in_wall', 'murphy_ultrasound',
    'ct_wall_thickened', 'ct_pericholecystic_fluid'
]

st.sidebar.header('Patient Input Features')

# Create a dictionary to store user input
user_input = {}

# Input widgets for numerical features
for feature in numeric_features:
    # Using a general min/max, ideally this would be based on training data distribution
    default_value = 0.0 # Placeholder, more robust would be to use mean/median from training data
    min_val = 0.0
    max_val = 500.0 # Adjust max based on feature scale

    if feature == 'age':
        default_value = 50.0
        max_val = 120.0
    elif feature == 'bmi':
        default_value = 25.0
        min_val = 10.0
        max_val = 60.0
    elif feature == 'onset_hours':
        default_value = 24.0
        max_val = 500.0
    elif feature == 'heart_rate':
        default_value = 80.0
        max_val = 200.0
    elif feature == 'systolic_bp':
        default_value = 120.0
        min_val = 50.0
        max_val = 250.0
    elif feature == 'diastolic_bp':
        default_value = 80.0
        min_val = 30.0
        max_val = 150.0
    elif feature == 'wbc':
        default_value = 10.0
        max_val = 50.0
    elif feature == 'neutrophil_pct':
        default_value = 70.0
        max_val = 100.0
    elif feature == 'nlr':
        default_value = 5.0
        max_val = 100.0
    elif feature == 'crp':
        default_value = 50.0
        max_val = 500.0
    elif feature == 'ast':
        default_value = 30.0
        max_val = 1000.0
    elif feature == 'alt':
        default_value = 30.0
        max_val = 1000.0
    elif feature == 'bilirubin_total':
        default_value = 1.0
        max_val = 50.0
    elif feature == 'creatinine':
        default_value = 1.0
        max_val = 20.0

    user_input[feature] = st.sidebar.number_input(
        f'\u2022 {feature.replace("_", " ").title()}',
        min_value=min_val,
        max_value=max_val,
        value=default_value,
        step=0.1
    )

# Input widgets for categorical features
for feature in categorical_features:
    user_input[feature] = st.sidebar.selectbox(
        f'\u2022 {feature.replace("_", " ").title()}',
        options=[0, 1] # Assuming binary (0 or 1) for all categorical features as seen in original data/pipeline
    )

# Convert user input to a DataFrame
input_df = pd.DataFrame([user_input])

st.subheader('User Input Features')
st.write(input_df)

# Make prediction (placeholder for now)
# if st.sidebar.button('Predict'):
#     prediction = model.predict(input_df)
#     prediction_proba = model.predict_proba(input_df)[:, 1]
#     st.subheader('Prediction')
#     st.write(f'Prediction: {prediction[0]}')
#     st.write(f'Prediction Probability (Necrosis): {prediction_proba[0]:.4f}')
# Make prediction
if st.sidebar.button('Predict'):
    # The model expects a DataFrame with columns matching the training features.
    # The preprocessor inside the pipeline will handle scaling and one-hot encoding.
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write("**Prediction: Necrosis Detected (1)**")
    else:
        st.write("**Prediction: No Necrosis (0)**")
    st.write(f'Prediction Probability of Necrosis: {prediction_proba[0]:.4f}')

    st.subheader('Explanation of Prediction Probability')
    if prediction_proba[0] > 0.5:
        st.info(f"The model predicts a higher chance of necrosis (probability: {prediction_proba[0]:.2%}).")
    else:
        st.info(f"The model predicts a lower chance of necrosis (probability: {prediction_proba[0]:.2%}).")

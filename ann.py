import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# --- Load artifacts ---
# IMPORTANT: avoid legacy auto-compile path on .h5
model = load_model("model.h5", compile=False)

with open("onehot_encoder_geo.pkl","rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("label_encoder_gender.pkl","rb") as f:
    label_encoder_gender = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Predictions")

# --- Inputs ---
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance", value=0.0, step=100.0, format="%.2f")
credit_score = st.number_input("Credit Score", value=650, step=1)
tenure = st.slider("Tenure", 0, 10, 1)
estimated_salary = st.number_input("Estimated Salary", value=0.0, step=100.0, format="%.2f")
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# --- Build feature row ---
df_num = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

geo_enc = onehot_encoder_geo.transform([[geography]]).toarray()
geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
df_geo = pd.DataFrame(geo_enc, columns=geo_cols)

X = pd.concat([df_num.reset_index(drop=True), df_geo], axis=1)

# If the scaler stores the original feature order, align to it:
if hasattr(scaler, "feature_names_in_"):
    missing = set(scaler.feature_names_in_) - set(X.columns)
    extra = set(X.columns) - set(scaler.feature_names_in_)
    if missing:
        st.error(f"Your input is missing columns expected by the scaler: {sorted(missing)}")
        st.stop()
    X = X[scaler.feature_names_in_]  # enforce exact order

# Scale and predict
X_scaled = scaler.transform(X)
prob = float(model.predict(X_scaled, verbose=0)[0][0])

st.write(f"**Churn probability:** {prob:.2f}")

# NOTE: If your model outputs P(churn), then prob > 0.5 means "will leave".
if prob > 0.5:
    st.success("Prediction: The customer **will leave** the bank.")
else:
    st.info("Prediction: The customer **will not leave** the bank.")





   
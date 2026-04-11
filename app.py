import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(page_title="Final Year Project", layout="wide")

st.title("Final Year Project 🚀")
st.write("App started successfully ✅")

# -------------------------------
# Dataset selection
# -------------------------------
dataset = st.selectbox("Select Dataset", [
    "benchmark_adult.csv",
    "finance_credit.csv",
    "healthcare_diabetes.csv"
])

# -------------------------------
# Load dataset safely
# -------------------------------
try:
    df = pd.read_csv(dataset)
    st.success(f"{dataset} loaded successfully ✅")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Basic preprocessing (safe)
    # -------------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    # Dummy model (safe for deployment)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 1),
        nn.Sigmoid()
    )

    # -------------------------------
    # Prediction (demo)
    # -------------------------------
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = (outputs > 0.5).int().numpy()

    st.subheader("Model Output (Demo)")
    st.write("Predictions generated successfully ✅")

    st.write(preds[:10])

except Exception as e:
    st.error(f"Error: {e}")
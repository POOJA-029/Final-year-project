import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(page_title="Final Year Project", layout="wide")

st.title("Final Year Project 🚀")
st.write("App started successfully ✅")

dataset = st.selectbox("Select Dataset", [
    "finance_credit.csv",
    "benchmark_adult.csv",
    "healthcare_diabetes.csv"
])

try:
    df = pd.read_csv(dataset)

    # 🔥 FIX: convert all to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    st.success("Dataset loaded successfully ✅")
    st.dataframe(df.head())

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values

    model = nn.Linear(X.shape[1], 1)

    with torch.no_grad():
        outputs = model(torch.tensor(X))
        preds = (outputs > 0.5).int().numpy()

    st.subheader("Predictions")
    st.write(preds[:10])

except Exception as e:
    st.error(f"Error: {e}")
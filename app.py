import streamlit as st
import pandas as pd

st.set_page_config(page_title="Final Year Project", layout="wide")

st.title("Final Year Project 🚀")

st.write("App started successfully ✅")

dataset = st.selectbox("Select Dataset", [
    "benchmark_adult.csv",
    "finance_credit.csv",
    "healthcare_diabetes.csv"
])

try:
    df = pd.read_csv(dataset)
    st.success(f"{dataset} loaded successfully ✅")
    st.dataframe(df.head())

except Exception as e:
    st.error(e)
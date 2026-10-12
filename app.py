import streamlit as st
import pandas as pd
import joblib

art = joblib.load("fraud_model.joblib")
cols = list(art["scaler"].feature_names_in_)

st.title("Credit Card Fraud Detector")
st.write("Upload a CSV with columns: Time, V1-V28, Amount (a 'Class' column is optional).")

f = st.file_uploader("CSV file", type="csv")
if f:
    df = pd.read_csv(f)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    p = art["model"].predict_proba(art["scaler"].transform(df[cols]))[:, 1]
    df["fraud_probability"] = p.round(4)
    df["flagged"] = p >= art["threshold"]
    st.metric("Flagged transactions", f"{int(df['flagged'].sum())} of {len(df)}")
    st.dataframe(df[df["flagged"]])
    if "Class" in df.columns:
        caught = int((df["flagged"] & (df["Class"] == 1)).sum())
        st.write(f"Actual frauds caught: {caught} of {int(df['Class'].sum())}")

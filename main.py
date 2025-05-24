import streamlit as st
import pandas as pd
import joblib
import numpy as np
from faker import Faker
import time
import random
import os

# Load model and scaler
if not os.path.exists("model/fraud_model.pkl") or not os.path.exists("model/scaler.pkl"):
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

model = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")

fake = Faker()

def generate_transaction():
    return {
        'amount': round(random.uniform(10, 1000), 2),
        'hour': random.randint(0, 23),
        'location': fake.city(),
        'card_holder': fake.name()
    }

def preprocess_transaction(txn):
    # Scale only amount and hour
    return scaler.transform([[txn['amount'], txn['hour']]])

def predict_fraud(txn):
    features = preprocess_transaction(txn)
    prediction = model.predict(features)[0]
    score = model.decision_function(features)[0]
    return 'Fraud' if prediction == -1 else 'Normal', score

st.set_page_config(page_title="Fraud Detection Dashboard")
st.title("📈 Real-Time Fraud Detection")

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Card Holder', 'Location', 'Amount ($)', 'Hour', 'Status', 'Score'])

placeholder = st.empty()

for _ in range(50):
    txn = generate_transaction()
    status, score = predict_fraud(txn)

    row = {
        'Card Holder': txn['card_holder'],
        'Location': txn['location'],
        'Amount ($)': txn['amount'],
        'Hour': txn['hour'],
        'Status': status,
        'Score': round(score, 3)
    }
    st.session_state.df.loc[len(st.session_state.df)] = row

    with placeholder.container():
        st.subheader("Live Transactions")
        st.dataframe(st.session_state.df.tail(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transactions", len(st.session_state.df))
        with col2:
            st.metric("Frauds Detected", (st.session_state.df['Status'] == 'Fraud').sum())

    time.sleep(1)

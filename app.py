import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Customer Churn Prediction App")

@st.cache_data
def load_and_train_model():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    x = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    x = pd.get_dummies(x, drop_first=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    xt = sc.fit_transform(x_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(xt, y_train)
    return model, sc, x.columns

model, sc, feature_columns = load_and_train_model()

st.subheader("Enter Customer Details")
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

if contract == "One year":
    if 'Contract_One year' in input_data.columns:
        input_data['Contract_One year'] = 1
elif contract == "Two year":
    if 'Contract_Two year' in input_data.columns:
        input_data['Contract_Two year'] = 1

if internet_service == "Fiber optic":
    if 'InternetService_Fiber optic' in input_data.columns:
        input_data['InternetService_Fiber optic'] = 1
elif internet_service == "No":
    if 'InternetService_No' in input_data.columns:
        input_data['InternetService_No'] = 1

if st.button("Predict Churn"):
    scaled_input = sc.transform(input_data)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)
    
    if prediction[0] == 1:
        st.error(f"High Risk of Churn! Probability: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"Customer is Stable. Churn Probability: {probability[0][1]*100:.2f}%")

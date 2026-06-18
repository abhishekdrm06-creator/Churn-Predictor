import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

df_raw = pd.read_csv("telco_customer_churn.csv")
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
df_raw = df_raw.dropna()
df_raw = df_raw.drop('customerID', axis=1)

st.set_page_config(page_title="AI Churn Predictor", layout="centered")
st.title("📡 Telco Customer Churn Predictor")
st.markdown("### System Online. AI Brain successfully connected.")
st.divider()

st.markdown("#### 👤 Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months connected)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 15.0, 120.0, 50.0)
    # Auto-calculate total charges for simplicity
    total_charges = tenure * monthly_charges 
    
with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if st.button("🔮 Predict Churn Risk"):
    
    baseline_customer = df_raw.iloc[0:1].copy()
    
    baseline_customer['tenure'] = tenure
    baseline_customer['MonthlyCharges'] = monthly_charges
    baseline_customer['TotalCharges'] = total_charges
    baseline_customer['Contract'] = contract
    baseline_customer['InternetService'] = internet
    
    combined_df = pd.concat([df_raw, baseline_customer], ignore_index=True)
    combined_dummies = pd.get_dummies(combined_df, drop_first=True)
    
    X_combined = combined_dummies.drop('Churn_Yes', axis=1)
    
    user_data = X_combined.iloc[[-1]]
    
    user_data_scaled = scaler.transform(user_data)
    
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)[0][1] # Get exact percentage
    
    st.divider()
    if prediction[0] == 1:
        st.error(f"🚨 HIGH RISK: This customer is likely to cancel! (Risk Factor: {probability*100:.1f}%)")
    else:
        st.success(f"✅ SAFE: This customer is secure. (Risk Factor: {probability*100:.1f}%)")
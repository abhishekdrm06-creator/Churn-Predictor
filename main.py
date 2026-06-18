import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("telco_customer_churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop('customerID', axis=1)
df_numeric = pd.get_dummies(df, drop_first=True)


X = df_numeric.drop('Churn_Yes', axis=1)
y = df_numeric['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)


print("AI is studying the data... please wait.")
model.fit(X_train_scaled, y_train)


predictions = model.predict(X_test_scaled)


score = accuracy_score(y_test, predictions)
print(f"\nFinal Exam Score (Accuracy): {score * 100:.2f}%")

import joblib


joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nExtraction Complete: AI Brain and Scaler permanently saved to your folder!")
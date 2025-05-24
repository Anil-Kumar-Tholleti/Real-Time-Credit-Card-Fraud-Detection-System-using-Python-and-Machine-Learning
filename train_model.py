import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/creditcard.csv")

# Create 'Hour' feature from 'Time' (Time is in seconds)
df['Hour'] = (df['Time'] // 3600) % 24

# Select only 'Amount' and 'Hour' as features
X = df[['Amount', 'Hour']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X_scaled)

# Save scaler and model
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(model, "model/fraud_model.pkl")

print("Training complete. Model and scaler saved to 'model/' folder.")

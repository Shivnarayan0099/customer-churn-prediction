import joblib
import pandas as pd

model = joblib.load("models/logistic_model.pkl")

sample = {
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines":0,
    "InternetService": 1,
    "OnlineSecurity": 1,
    "OnlineBackup": 0,
    "DeviceProtection": 1,
    "TechSupport": 1,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 1,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 70.5,
    "TotalCharges": 845.5
}

df = pd.DataFrame([sample])
prediction = model.predict(df)
if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Not Churn")

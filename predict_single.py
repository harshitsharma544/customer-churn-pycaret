from pycaret.classification import load_model, predict_model
import pandas as pd

# Load trained model
print("\n🔄 Loading trained model...")
model = load_model("customer_churn_model")
print("✔ Model Loaded Successfully!\n")

# New customer data
new_customer = {
    'customerID': '0001',   # 🔥 REQUIRED — FIXES ERROR
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 75.45,
    'TotalCharges': 325.50
}

# Convert to dataframe
new_df = pd.DataFrame([new_customer])

# Predict
result = predict_model(model, new_df)

print("\n🔍 Prediction for new customer:")
print(result[['prediction_label', 'prediction_score']])

input("\n👉 Press Enter to exit...")

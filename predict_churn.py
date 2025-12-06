from pycaret.classification import load_model, predict_model
import pandas as pd

# Load trained model
model = load_model("customer_churn_model")

print("✔ Model Loaded Successfully!\n")

# Load dataset again (or new customer data)
data = pd.read_csv("telco.csv")

# Predict on first 5 rows
predictions = predict_model(model, data.head())

print("🔍 Sample Predictions:")
print(predictions[['Churn', 'prediction_label', 'prediction_score']])

input("\n👉 Press Enter to exit...")

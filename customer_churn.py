from pycaret.classification import *
import pandas as pd

print("\n==================== Customer Churn Model Training ====================\n")

# Load dataset
print("📌 Loading dataset...")
data = pd.read_csv("telco.csv")
print(f"✔ Dataset loaded successfully! Shape: {data.shape}\n")

# Setup experiment
print("📌 Initializing PyCaret setup (this may take some time)...")
exp = setup(
    data=data,
    target='Churn',
    session_id=123,
    verbose=True   # silent removed because PyCaret 3 doesn't support it
)

# Compare models
print("\n📌 Comparing models and selecting best...")
best_model = compare_models()
print("✔ Best model selected!\n")

# Finalize the best model
print("📌 Finalizing the best model...")
final_model = finalize_model(best_model)
print("✔ Model finalized successfully!\n")

# Save model
print("📌 Saving the model...")
save_model(final_model, 'customer_churn_model')
print("✔ Model saved successfully as 'customer_churn_model.pkl' 🎉")

print("\n==================== Training Completed ====================\n")
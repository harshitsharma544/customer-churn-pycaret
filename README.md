📌 Customer Churn Prediction using PyCaret
🚀 Project Overview

This project predicts whether a telecom customer is likely to churn.
It uses automated machine learning (AutoML) with PyCaret to find the best model and make predictions.

📂 Project Files
File	Description
customer_churn.py	Trains churn model & saves it
predict_churn.py	Makes batch predictions
predict_single.py	Predicts churn for a single customer
customer_churn_model.pkl	Saved machine learning model
telco.csv	Dataset
logs.log	Training logs
🔧 Tech Stack

✔ Python
✔ Pandas
✔ PyCaret
✔ Machine Learning
✔ VS Code

🔍 Features

✔ Automatic model selection
✔ Churn probability prediction
✔ Single & batch prediction support
✔ Easy-to-run scripts

📊 Output Example
🔹 Sample Predictions
Churn	prediction_label	prediction_score
0	No	0.60
1	No	1.00
2	Yes	0.60
3	No	1.00
4	Yes	0.80
🔹 Single Customer Prediction
prediction_label   Yes  
prediction_score   0.6

▶️ How to Run
1️⃣ Train Model
python customer_churn.py

2️⃣ Predict in bulk
python predict_churn.py

3️⃣ Predict single customer
python predict_single.py

💡 Future Enhancements

🔹 Build UI using Streamlit or Flask
🔹 Deploy model on cloud (Render / HuggingFace / AWS)

📬 Author

👤 Harshit Sharma

📌 Feel free to ⭐ star this repository or connect with me!

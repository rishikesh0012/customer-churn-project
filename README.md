 Customer Churn Prediction Dashboard

An interactive **Customer Churn Prediction system** built with **Python, scikit-learn, and Streamlit**.  
It predicts the likelihood of a customer leaving a telecom company and shows **visual insights**.

---

## Features

- Predict customer churn probability using **Logistic Regression**
- Handle missing numeric values automatically
- Interactive **Streamlit dashboard** with:
  - KPIs (Total Customers, High-Risk Customers, Avg. Charges)
  - Churn Probability distribution
  - Filter high-risk customers and download predictions
  - Charts for contracts, payments, tenure, and monthly charges

---

## Project Structure

customer-churn-project/
├── app.py
├── retrain_model.py
├── churn_model.pkl
├── Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
└── venv/

---

## How to Run

1. Activate virtual environment:
```bash
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
Open browser at http://localhost:8501 and upload your CSV.
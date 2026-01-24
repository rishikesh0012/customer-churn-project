## **Customer Churn Prediction Dashboard**

```markdown
# Customer Churn Prediction Dashboard

A machine learning system designed to **predict customer churn probability**, enabling proactive retention strategies.

---

## 🔍 Problem Statement
Customer churn directly impacts revenue and growth. Businesses need early signals to:
- Identify at-risk customers
- Take preventive retention actions
- Reduce customer acquisition costs

---

## 💡 Solution
This project builds a **Logistic Regression–based churn prediction model** and visualizes churn risk through an interactive dashboard.

---

## 🧠 Machine Learning Details
- **Algorithm:** Logistic Regression
- **Feature Engineering:** Data normalization, categorical encoding
- **Evaluation Metric:** ROC-AUC
- **Performance:** ROC-AUC = 0.83

---

## 🏗️ System Flow
1. Customer data ingestion  
2. Data preprocessing & feature engineering  
3. Model training & evaluation  
4. Churn probability prediction  
5. Dashboard visualization  

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Streamlit

---

## 📊 Output
- Individual customer churn probability
- Business KPIs
- Batch prediction via CSV upload

---

## ▶️ How to Run
```bash
git clone https://github.com/rishikesh0012/customer-churn-project.git
cd customer-churn-project
pip install -r requirements.txt
streamlit run app.py

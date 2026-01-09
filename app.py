import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
pipeline = joblib.load("churn_model.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Overview", "🔮 Predictions", "📊 Insights", "ℹ️ About"])
uploaded_file = st.sidebar.file_uploader("📂 Upload Telco Customer CSV", type=["csv"])

# ---------------- PREPROCESSING ----------------
@st.cache_data
def preprocess_data(df):
    df = df.copy()
    
    # Convert numeric columns
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Fill missing numeric values
    imputer = SimpleImputer(strategy="median")
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Separate target if exists
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes":1, "No":0})
        X = df.drop(["customerID", "Churn"], axis=1, errors="ignore")
    else:
        y = None
        X = df.drop(["customerID"], axis=1, errors="ignore")
    
    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)
    
    # Align columns with trained model
    X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)
    
    return X, y, df

# ---------------- RUN APP ----------------
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    X, y, df_raw = preprocess_data(df_raw)

    # Make predictions
    churn_prob = pipeline.predict_proba(X)[:, 1]
    
    # Add to original df safely
    df_raw["Churn Probability"] = churn_prob
    df_raw["Risk Segment"] = df_raw["Churn Probability"].apply(lambda x: "High Risk" if x > 0.7 else "Low Risk")

# ---------------- PAGES ----------------
if page == "🏠 Overview":
    st.title("📉 Customer Churn Dashboard")
    st.markdown("Upload your customer dataset to see predictions and insights.")
    
    if uploaded_file:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df_raw))
        col2.metric("High Risk Customers", (df_raw["Risk Segment"]=="High Risk").sum())
        col3.metric("Average Monthly Charges", f"${df_raw['MonthlyCharges'].mean():.2f}")
        col4.metric("Average Churn Probability", f"{df_raw['Churn Probability'].mean():.2f}")
        
        st.markdown("### 🚨 Churn Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_raw["Churn Probability"], bins=10, ax=ax)
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Customer Count")
        st.pyplot(fig)
    else:
        st.info("⬅ Upload a dataset to see overview")

elif page == "🔮 Predictions":
    st.title("🔮 Customer Churn Predictions")
    if uploaded_file:
        threshold = st.slider("Show customers with churn probability above:", 0.0, 1.0, 0.7)
        filtered = df_raw[df_raw["Churn Probability"] >= threshold]
        st.dataframe(filtered)
        st.download_button("📥 Download Predictions", filtered.to_csv(index=False), file_name="churn_predictions.csv")
    else:
        st.info("⬅ Upload a dataset first")

elif page == "📊 Insights":
    st.title("📊 Churn Insights")
    if uploaded_file:
        tab1, tab2, tab3 = st.tabs(["📄 Contracts", "💳 Payments", "💰 Charges & Tenure"])
        
        with tab1:
            fig, ax = plt.subplots()
            sns.boxplot(x="Contract", y="Churn Probability", data=df_raw, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.boxplot(x="PaymentMethod", y="Churn Probability", data=df_raw, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.scatterplot(x="MonthlyCharges", y="Churn Probability", data=df_raw, ax=ax)
                ax.set_xlabel("Monthly Charges")
                ax.set_ylabel("Churn Probability")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots()
                sns.scatterplot(x="tenure", y="Churn Probability", data=df_raw, ax=ax)
                ax.set_xlabel("Tenure (Months)")
                ax.set_ylabel("Churn Probability")
                st.pyplot(fig)
    else:
        st.info("⬅ Upload a dataset first")

else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Customer Churn Prediction System**
    - ML Model: Logistic Regression
    - Pipeline: Safe imputation + scaling
    - Interactive Streamlit dashboard with predictions, KPIs, and charts
    """)


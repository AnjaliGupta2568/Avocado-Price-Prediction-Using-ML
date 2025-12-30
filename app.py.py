import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Avocado Price Predictor v2",
    layout="wide"
)

st.title("🥑 Avocado Price Prediction (ML Project)")
st.write("Upload the avocado dataset to perform EDA, train models, and predict prices.")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
file = st.file_uploader("📂 Upload CSV File", type="csv")

if file is None:
    st.info("Please upload the avocado dataset to continue.")
    st.stop()

df = pd.read_csv(file)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Controls")
section = st.sidebar.radio(
    "Choose Section",
    ["EDA", "Train Model", "Model Evaluation", "Insights"]
)

model_name = st.sidebar.selectbox(
    "Select Regression Model",
    ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]
)

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------
data = df.copy()
data = pd.get_dummies(data, columns=["type", "region"], drop_first=True)

X = data.drop(["AveragePrice", "Date"], axis=1)
y = data["AveragePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
def get_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Ridge":
        return Ridge(alpha=1.0)
    elif name == "Lasso":
        return Lasso(alpha=0.01)
    elif name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42)
    else:
        return RandomForestRegressor(n_estimators=100, random_state=42)

model = get_model(model_name)

# --------------------------------------------------
# EDA SECTION
# --------------------------------------------------
if section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Shape & Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.write(df.isnull().sum())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["AveragePrice"], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Price by Type")
        fig, ax = plt.subplots()
        sns.boxplot(x="type", y="AveragePrice", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Yearly Average Price Trend")
    fig, ax = plt.subplots()
    df.groupby("year")["AveragePrice"].mean().plot(ax=ax)
    ax.set_ylabel("Average Price")
    st.pyplot(fig)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
elif section == "Train Model":
    st.header("🤖 Model Training")

    if model_name in ["Linear Regression", "Ridge", "Lasso"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    st.success(f"Model **{model_name}** trained successfully!")

    result_df = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": predictions
    })

    st.subheader("Prediction Sample")
    st.dataframe(result_df.head(10))

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
elif section == "Model Evaluation":
    st.header("📈 Model Evaluation Metrics")

    if model_name in ["Linear Regression", "Ridge", "Lasso"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(mae, 3))
    col2.metric("MSE", round(mse, 3))
    col3.metric("R² Score", round(r2, 3))

    st.subheader("Residual Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=y_test - y_pred, ax=ax)
    ax.axhline(0, color="red")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------
elif section == "Insights":
    st.header("🧠 Insights & Conclusion")

    st.markdown("""
### 🔍 Observations
- Average price is influenced by **region, year, and total volume**
- Conventional and organic avocados show different pricing behavior

### 🤖 Model Learnings
- Linear & Ridge models perform well for simple trends
- Random Forest captures non-linear relationships better

### 📊 Business Value
- Helps retailers forecast prices
- Supports supply chain planning
- Useful for seasonal demand analysis

### ✅ Conclusion
This project demonstrates:
- Data preprocessing
- Feature engineering
- Multiple ML models
- Evaluation & interpretation
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("👩‍💻 Developed by **Anjali Gupta**")

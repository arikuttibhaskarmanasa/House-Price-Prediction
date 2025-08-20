import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------------
# APP TITLE
# -----------------------------
st.set_page_config(page_title="Advanced House Price Prediction", layout="wide")
st.title("🏡 Advanced House Price Prediction Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Replace with Kaggle dataset
    return df

df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# EDA SECTION
# -----------------------------
st.subheader("📈 Data Exploration")
col1, col2 = st.columns(2)

with col1:
    st.write("Basic Statistics")
    st.write(df.describe())

with col2:
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    st.pyplot(fig)

# -----------------------------
# FEATURES & TARGET
# -----------------------------
target = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])

X = df[features]
y = df[target]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN & COMPARE MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

st.subheader("🤖 Model Training & Evaluation")
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
    results[name] = np.mean(scores)

result_df = pd.DataFrame.from_dict(results, orient="index", columns=["R² Score"]).sort_values(by="R² Score", ascending=False)
st.bar_chart(result_df)

best_model_name = result_df.index[0]
st.success(f"Best Model: {best_model_name} with R² = {results[best_model_name]:.3f}")

# -----------------------------
# TRAIN FINAL MODEL
# -----------------------------
final_model = models[best_model_name]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

st.subheader("📉 Final Model Performance")
st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# -----------------------------
# USER PREDICTION
# -----------------------------
st.subheader("🔍 Predict House Price")

user_input = []
for f in features:
    val = st.number_input(f"Enter {f}", float(df[f].min()), float(df[f].max()), float(df[f].mean()))
    user_input.append(val)

if st.button("Predict Price"):
    input_scaled = scaler.transform([user_input])
    prediction = final_model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Price: {prediction:,.2f}")

# -----------------------------
# EXPORT MODEL
# -----------------------------
if st.button("Save Model"):
    joblib.dump(final_model, f"models/{best_model_name}.pkl")
    st.success("Model saved successfully!")

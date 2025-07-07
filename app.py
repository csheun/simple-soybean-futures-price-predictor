import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import time

# === Page config ===
st.set_page_config(page_title="Soybean Price Predictor", page_icon="🌱", layout="wide")

# === Title and intro ===
st.title("🌱 Soybean Futures Price Predictor")
st.markdown("""
Use machine learning to forecast next-day **soybean futures prices** using historical data from **January 1, 2022** to present.  
The model leverages previous closing prices and moving averages as features.  

Adjust model choice and hyperparameters in the sidebar, then click **Train Model** to see predictions and performance metrics.
""")

# === Load & cache data ===
@st.cache_data
def load_data():
    df = yf.download("ZS=F", start="2022-01-01")[['Close']].dropna()
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['MA_3'] = df['Close'].rolling(3).mean()
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['Target'] = df['Close'].shift(-1)
    return df.dropna()

df = load_data()

# === Sidebar controls ===
st.sidebar.header("🔧 Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])
n_estimators = st.sidebar.slider("Number of Trees", 10, 300, 100)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 20, 5)
train_split = st.sidebar.slider("Train/Test Split %", 50, 90, 80)
run_button = st.sidebar.button("🚀 Train Model")

# === Prepare data ===
features = ['Close_lag1', 'Close_lag2', 'MA_3', 'MA_7']
X = df[features]
y = df['Target']
split_idx = int(len(df) * train_split / 100)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# === Clear old model results if parameters change but not retrained ===
if not run_button:
    st.session_state.pop("model", None)
    st.session_state.pop("preds", None)
    st.session_state.pop("rmse", None)
    st.session_state.pop("y_test", None)

# === Train model when button pressed ===
if run_button:
    with st.spinner("Training model... please wait"):
        progress = st.progress(0)
        for pct in range(0, 100, 20):
            time.sleep(0.2)
            progress.progress(pct + 20)

        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, verbosity=0)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)

        # Save results in session state
        st.session_state["model"] = model
        st.session_state["preds"] = preds
        st.session_state["rmse"] = rmse
        st.session_state["y_test"] = y_test

# === Show results only if trained and lengths match ===
if "model" in st.session_state and \
   len(st.session_state["preds"]) == len(st.session_state["y_test"]):

    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{st.session_state['rmse']:.2f}")
    col2.metric("Training Rows", f"{len(X_train)}")
    col3.metric("Testing Rows", f"{len(X_test)}")

    tab1, tab2 = st.tabs(["📈 Predictions", "🧠 Feature Importance"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(st.session_state["y_test"].index, st.session_state["y_test"].values, label="Actual", linewidth=2)
        ax.plot(st.session_state["y_test"].index, st.session_state["preds"], label="Predicted", linestyle="--")
        ax.set_ylabel("Soybean Price ($)")
        ax.set_xlabel("Date")
        ax.set_title("Actual vs Predicted Soybean Prices")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        importances = st.session_state["model"].feature_importances_
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))

else:
    st.info("⬅️ Set parameters and click **Train Model** to see results.")

# === Optional raw data viewer ===
with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df.tail(20))

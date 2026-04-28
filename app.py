# =====================================
# QuantGuard - Real-Time Version
# Live + Dashboard + Predictions
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="QuantGuard", layout="wide")  # PAGE SETTINGS

st.title("📊 QuantGuard - Smart Market Predictor")  # STREAMLIT TITLE

# ---------------------------------
# STOCK SELECT
# ---------------------------------
stock = st.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META"]
)

# ---------------------------------
# LOAD LIVE DATA
# ---------------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="5y", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    return df

df = load_data(stock)

# ---------------------------------
# FEATURES
# ---------------------------------
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_50"] = df["Close"].rolling(50).mean()
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
df["Volatility"] = df["Close"].rolling(10).std()
df["Returns"] = df["Close"].pct_change()
df["Crossover"] = np.where(df["MA_10"] > df["MA_50"], 1, 0)

# Target
df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

df.dropna(inplace=True)

# ---------------------------------
# MACHINE LEARNING
# ---------------------------------
features = ["MA_10", "MA_50", "RSI", "Volatility", "Returns", "Crossover"]

X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

# ---------------------------------
# TODAY STATUS
# ---------------------------------
latest_price = df["Close"].iloc[-1]
prev_price = df["Close"].iloc[-2]
change = latest_price - prev_price
percent = (change / prev_price) * 100

# ---------------------------------
# FUTURE PREDICTION
# ---------------------------------
latest = X.tail(1)
future = model.predict(latest)

# ---------------------------------
# TOP CARDS
# ---------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Price", f"${latest_price:.2f}")

with col2:
    st.metric("Daily Change", f"{change:.2f}", f"{percent:.2f}%")

with col3:
    st.metric("Model Accuracy", f"{accuracy:.2f}")

# ---------------------------------
# NEXT DAY PREDICTION
# ---------------------------------
st.subheader("🔮 Next Trading Day Prediction")

if future[0] == 1:
    st.success("📈 Market Likely to Go UP Tomorrow")
else:
    st.error("📉 Market Likely to Go DOWN Tomorrow")

# ---------------------------------
# PREVIOUS 5 DAYS
# ---------------------------------
st.subheader("📅 Previous 5 Trading Days")

last5 = df[["Date", "Close"]].tail(5)
st.dataframe(last5, use_container_width=True)

# ---------------------------------
# NEXT 5 DAYS PREDICTION
# ---------------------------------
st.subheader("📈 Future 5 Trading Days Prediction")

last_date = df["Date"].iloc[-1]

future_data = []
temp = latest.copy()

for i in range(1, 6):
    pred = model.predict(temp)[0]

    if pred == 1:
        result = "UP 📈"
    else:
        result = "DOWN 📉"

    future_date = last_date + pd.Timedelta(days=i)

    # skip weekends (simple fix)
    while future_date.weekday() >= 5:
        future_date += pd.Timedelta(days=1)

    future_data.append([future_date.date(), result])

future_df = pd.DataFrame(future_data, columns=["Date", "Prediction"])
st.dataframe(future_df, use_container_width=True)

# ---------------------------------
# DASHBOARD
# ---------------------------------

st.subheader("📊 Dashboard")

# ================= ROW 1 =================
st.markdown("## Price & Trend Analysis")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Closing Price")

    fig1, ax1 = plt.subplots(figsize=(6, 3))  # 🔥 SMALL SIZE FIX
    ax1.plot(df["Date"], df["Close"], linewidth=1.2)
    ax1.set_title("Closing Price Trend")
    ax1.grid(True, alpha=0.3)

    st.pyplot(fig1)

with col2:
    st.markdown("### Moving Averages")

    fig2, ax2 = plt.subplots(figsize=(6, 3))  # 🔥 SMALL SIZE FIX
    ax2.plot(df["Date"], df["Close"], label="Close", linewidth=1)
    ax2.plot(df["Date"], df["MA_10"], label="MA 10", linewidth=1)
    ax2.plot(df["Date"], df["MA_50"], label="MA 50", linewidth=1)

    ax2.set_title("Moving Average Analysis")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)
# ================= ROW 2 =================
st.markdown("## Market Indicators")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📉 RSI Indicator")
    st.line_chart(df.set_index("Date")["RSI"], use_container_width=True)

with col2:
    st.markdown("### 📊 Volatility (Risk Measure)")
    st.line_chart(df.set_index("Date")["Volatility"], use_container_width=True)

st.divider()

# ================= ROW 3 =================
st.markdown("## Trading Signals")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📊 Returns (%)")
    st.line_chart(df.set_index("Date")["Returns"], use_container_width=True)

with col2:
    st.markdown("### MA Crossover Signal")

    fig3, ax3 = plt.subplots(figsize=(6, 3))

    ax3.plot(df["Date"], df["Crossover"], color="purple", linewidth=1)
    ax3.set_title("Bullish (1) / Bearish (0) Signal")
    ax3.set_yticks([0, 1])
    ax3.grid(True, alpha=0.3)

    st.pyplot(fig3)

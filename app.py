import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(page_title="Portfolio Risk & Optimization Dashboard", layout="wide")
st.title("📊 Portfolio Risk & Optimization Dashboard")
st.caption("10 years of data | 11 US stocks | SQL + Python + ML + Optimization")

# ---- Load data ----
conn = sqlite3.connect("data/portfolio.db")
prices = pd.read_sql("SELECT * FROM daily_prices", conn)
risk = pd.read_sql("SELECT * FROM risk_metrics", conn)
models = pd.read_sql("SELECT * FROM model_results", conn)
conn.close()

prices["Date"] = pd.to_datetime(prices["Date"])

# ---- Sidebar: stock selector ----
st.sidebar.header("Filters")
selected_tickers = st.sidebar.multiselect(
    "Select stocks to view",
    options=risk["Ticker"].tolist(),
    default=risk["Ticker"].tolist()
)

# ---- Section 1: Risk Metrics Table ----
st.subheader("Risk & Return Metrics")
st.dataframe(
    risk[risk["Ticker"].isin(selected_tickers)].sort_values("Sharpe_Ratio", ascending=False),
    use_container_width=True
)

# ---- Section 2: Price chart ----
st.subheader("Price History")
filtered_prices = prices[prices["Ticker"].isin(selected_tickers)]
fig_price = px.line(filtered_prices, x="Date", y="Close", color="Ticker",
                     title="Closing Price Over Time")
st.plotly_chart(fig_price, use_container_width=True)

# ---- Section 3: Risk vs Return scatter (proxy for efficient frontier view) ----
st.subheader("Risk vs Return")
fig_scatter = px.scatter(
    risk, x="Annual_Volatility", y="Annual_Return", text="Ticker",
    size="Sharpe_Ratio", color="Sharpe_Ratio", color_continuous_scale="Viridis",
    title="Annual Volatility vs Annual Return (bubble size = Sharpe Ratio)"
)
fig_scatter.update_traces(textposition="top center")
st.plotly_chart(fig_scatter, use_container_width=True)

# ---- Section 4: Model performance ----
st.subheader("Forecasting Model Performance (vs Naive Baseline)")
st.dataframe(models, use_container_width=True)

# ---- Section 5: Optimized Portfolio Allocation ----
st.subheader("Optimized Portfolio Allocation (Max Sharpe)")
# These weights come from Task 5 - paste your actual output here
optimal_weights = {
    "NVDA": 42.2, "CAT": 22.3, "JNJ": 17.7, "AAPL": 12.1, "KO": 5.6
}
fig_pie = px.pie(
    values=list(optimal_weights.values()),
    names=list(optimal_weights.keys()),
    title="Recommended Allocation"
)
st.plotly_chart(fig_pie, use_container_width=True)

# ---- Section 6: AI-generated summary ----
import time

if st.button("Generate AI Summary"):
    with st.spinner("Generating summary..."):
       # Works both locally (.env) and on Streamlit Cloud (secrets)
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        risk_summary = risk[["Ticker", "Annual_Return", "Annual_Volatility", "Sharpe_Ratio"]].to_string(index=False)
        weights_summary = ", ".join([f"{k}: {v}%" for k, v in optimal_weights.items()])

        prompt = f"""You are a financial analyst assistant. Based on the following data, write a concise
3-4 sentence plain-English summary of this optimized stock portfolio for a non-technical reader.
Mention the top holding, overall risk level, and one diversification insight.

Risk metrics per stock:
{risk_summary}

Optimized portfolio weights (max Sharpe ratio): {weights_summary}
"""
        # Try a couple of models, with short retries, before giving up
        models_to_try = ["gemini-2.0-flash-lite", "gemini-flash-latest"]
        success = False

        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    st.success(response.text)
                    st.caption(f"(Generated using {model_name})")
                    success = True
                    break
                except Exception as e:
                    if attempt == 0:
                        time.sleep(3)  # brief pause before retrying
                    continue
            if success:
                break

        if not success:
            st.warning("Google's free-tier AI models are experiencing high demand right now. This is common during peak hours — please try again in a few minutes.")
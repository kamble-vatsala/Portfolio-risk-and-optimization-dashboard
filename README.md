# 📊 Portfolio Risk & Optimization Dashboard

An end-to-end quantitative finance project covering SQL, Python, statistics, machine learning, and portfolio optimization — built to analyze 11 major US stocks, forecast returns, and generate an optimal portfolio allocation.

**🔗 Live app:** https://portfolio-risk-and-optimization-dashboard-vatsala.streamlit.app/

## Overview

This project pulls 10 years of daily price data for 11 US stocks across sectors (tech, finance, healthcare, consumer, energy, industrials), stores it in a SQL database, engineers features, computes risk metrics, builds a forecasting model, runs Markowitz portfolio optimization, and presents everything in an interactive dashboard — including an AI-generated plain-English summary of the results.

## Tech Stack

- **Data & Storage:** Python, yfinance, SQLite
- **Analysis:** pandas, NumPy
- **Machine Learning:** scikit-learn (Linear Regression, Random Forest)
- **Optimization:** PyPortfolioOpt (Markowitz mean-variance optimization)
- **Dashboard:** Streamlit, Plotly
- **AI Integration:** Google Gemini API

## Pipeline

1. **Data Collection** — Pulled 10 years of daily OHLCV data for AAPL, MSFT, JPM, JNJ, KO, XOM, PG, DIS, CAT, NVDA, and GOOGL via `yfinance`, stored in SQLite.
2. **Feature Engineering** — Computed daily returns, 20-day rolling volatility, and 50/200-day moving averages per stock.
3. **Risk Metrics** — Calculated annualized return/volatility, Sharpe ratio, maximum drawdown, and 95% historical Value at Risk (VaR) for each stock.
4. **Forecasting Model** — Built a Linear Regression model on lagged return/volatility/trend features to predict next-day returns, evaluated against a naive baseline and a Random Forest model using a time-based train/test split (no shuffling, to avoid lookahead bias).
5. **Portfolio Optimization** — Applied Markowitz mean-variance optimization to find the max-Sharpe-ratio portfolio allocation across all 11 stocks.
6. **Dashboard** — Built an interactive Streamlit app with filterable risk tables, price charts, a risk-vs-return visualization, model comparison, the optimized allocation, and an on-demand AI-generated natural-language summary via the Gemini API.

## Key Findings

- **NVDA** had the strongest risk-adjusted performance (Sharpe ratio 1.64) despite the highest volatility and a -66% max drawdown, while defensive stocks like **JNJ, KO, and PG** showed low volatility and shallow drawdowns — consistent with real-world sector behavior.
- The forecasting model consistently beat a naive "tomorrow equals today" baseline on RMSE across all 11 stocks, even though R² was mostly negative — a result consistent with the Efficient Market Hypothesis, and a reminder that R² and practical predictive value aren't the same thing.
- Random Forest performed comparably to Linear Regression, suggesting the modest signal in the features is largely linear — added model complexity wasn't justified here.
- The optimizer allocated meaningfully to **JNJ (17.7%)** despite its low standalone Sharpe ratio, illustrating that diversification value — not just individual asset quality — drives Markowitz optimization. The final portfolio (NVDA 42%, CAT 22%, JNJ 18%, AAPL 12%, KO 6%) achieved a Sharpe ratio of 1.29.

## Run Locally

```bash
git clone https://github.com/kamble-vatsala/Portfolio-risk-and-optimization-dashboard
cd Portfolio-risk-and-optimization-dashboard
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Gemini API key from [Google AI Studio](https://aistudio.google.com) — add it to a `.env` file as `GEMINI_API_KEY=your_key_here`.

## Project Structure

```
├── fetch_data.py          # Pulls stock data into SQLite
├── clean_features.py      # Cleans data, engineers features
├── risk_metrics.py        # Computes risk/return metrics
├── forecast_model.py      # Linear Regression + Random Forest forecasting
├── optimize_portfolio.py  # Markowitz portfolio optimization
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── data/portfolio.db      # SQLite database
```

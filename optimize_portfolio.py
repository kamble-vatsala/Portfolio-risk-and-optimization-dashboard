import pandas as pd
import sqlite3
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

conn = sqlite3.connect("data/portfolio.db")
df = pd.read_sql("SELECT Date, Close, Ticker FROM daily_prices", conn)
conn.close()

df["Date"] = pd.to_datetime(df["Date"])

# Pivot so each ticker is a column of closing prices, indexed by date
prices = df.pivot(index="Date", columns="Ticker", values="Close")
prices = prices.dropna()  # keep only dates where ALL 11 stocks have data

# Expected returns (annualized, using historical mean) and covariance matrix
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Optimize for max Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate=0.04)
cleaned_weights = ef.clean_weights()

print("Optimal Portfolio Weights (Max Sharpe):")
for ticker, weight in cleaned_weights.items():
    if weight > 0:
        print(f"  {ticker}: {weight*100:.1f}%")

perf = ef.portfolio_performance(verbose=True, risk_free_rate=0.04)

# Example: how to actually allocate a real amount, e.g. $100,000
latest_prices = get_latest_prices(prices)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da.lp_portfolio()
print("\nSample allocation for $100,000:")
print(allocation)
print(f"Leftover cash: ${leftover:.2f}")
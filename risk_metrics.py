import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect("data/portfolio.db")
df = pd.read_sql("SELECT * FROM features", conn)
df["Date"] = pd.to_datetime(df["Date"])

results = []

for ticker, group in df.groupby("Ticker"):
    returns = group["Daily_Return"].dropna()

    # Annualized return (compounding daily returns, ~252 trading days/year)
    annual_return = (1 + returns.mean()) ** 252 - 1

    # Annualized volatility
    annual_volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming ~4% risk-free rate, a reasonable current approximation)
    risk_free_rate = 0.04
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    # Max drawdown: worst peak-to-trough decline
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Historical VaR at 95% confidence (worst expected daily loss, 95% of the time)
    var_95 = np.percentile(returns, 5)

    results.append({
        "Ticker": ticker,
        "Annual_Return": annual_return,
        "Annual_Volatility": annual_volatility,
        "Sharpe_Ratio": sharpe_ratio,
        "Max_Drawdown": max_drawdown,
        "VaR_95_Daily": var_95
    })

risk_df = pd.DataFrame(results).sort_values("Sharpe_Ratio", ascending=False)
risk_df.to_sql("risk_metrics", conn, if_exists="replace", index=False)
conn.close()

print(risk_df.round(4))
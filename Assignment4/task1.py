import yfinance as yf 
import pandas as pd 
import torch 
from sklearn.preprocessing import MinMaxScaler

tickers = ["AAPL", "TSLA", "GOOGL", "PFE"]
start = "2020-01-01"
end = "2022-01-01"

dfs = {}
for ticker in tickers:
    dat = yf.Ticker(ticker)
    df = dat.history(start=start,end=end)
    dfs[ticker] = df["Close"]

prices = pd.DataFrame(dfs).dropna()


scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

prices_tensor = torch.tensor(prices_scaled, dtype=torch.float32)
print(prices_tensor.shape)
print(prices_tensor[:5])

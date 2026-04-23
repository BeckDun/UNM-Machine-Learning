import yfinance as yf 
import pandas as pd 
import torch 
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import task2

def make_sequences(series, M, N):
    X, y = [], []
    for i in range(len(series) - M - N + 1):
        X.append(series[i:i+M])
        y.append(series[i+M:i+M+N])
    return torch.stack(X), torch.stack(y)

def pre_process_data():
    tickers = ["AAPL", "TSLA", "GOOGL", "PFE"]
    start = "2020-01-01"
    end = "2022-01-01"
    M = 60
    N = 1

    dfs = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        dfs[ticker] = df["Close"].squeeze()
        # I was running into a limiting error with yfinance so I wait 5 seconds between request. (You might also have to upgrade yfinance)
        time.sleep(5)

    prices = pd.DataFrame(dfs).dropna()

    scalers = {}
    prices_scaled = prices.copy()
    for ticker in tickers:
        sc = MinMaxScaler()
        prices_scaled[ticker] = sc.fit_transform(prices[[ticker]])
        scalers[ticker] = sc

    datasets = {}
    for ticker in tickers:
        series = torch.tensor(prices_scaled[ticker].values, dtype=torch.float32)
        X, y = make_sequences(series, M, N)

        idx = torch.randperm(len(X))
        X, y = X[idx], y[idx]

        n_train = int(len(X) * 0.8)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        datasets[ticker] = {
            "X_train": X_train.unsqueeze(-1),
            "y_train": y_train,
            "X_test": X_test.unsqueeze(-1),
            "y_test": y_test,
            "scaler": scalers[ticker],
        }
    return datasets
    

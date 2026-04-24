import yfinance as yf 
import pandas as pd 
import numpy as np
import torch 
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import task2

np.random.seed(1)
torch.manual_seed(1)

# download price data
tickers = ["KO", "WMT", "SPY", "ED"]
start, end = "2020-01-01", "2022-01-01"

dfs = {}
for t in tickers:
    df = yf.download(t, start=start, end=end, progress=False)
    dfs[t] = df["Close"].squeeze()
    print(f"downloaded {t}")
    time.sleep(3)

prices = pd.DataFrame(dfs).dropna()

N = 60
M = 1


def make_sequences(series, lookback, horizon):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return (
        np.array(X).reshape(-1, lookback).astype(np.float32),
        np.array(y).reshape(-1, 1).astype(np.float32),
    )


datasets = {}
for t in tickers:
    vals = prices[t].values
    X, y = make_sequences(vals, N, M)

    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.8)

    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()

    X_tr = sc_X.fit_transform(X[:split]).reshape(-1, N, 1).astype(np.float32)
    X_te = sc_X.transform(X[split:]).reshape(-1, N, 1).astype(np.float32)
    y_tr = sc_y.fit_transform(y[:split]).astype(np.float32)
    y_te = sc_y.transform(y[split:]).astype(np.float32)

    datasets[t] = {
        "X_train": torch.from_numpy(X_tr),
        "y_train": torch.from_numpy(y_tr),
        "X_test": torch.from_numpy(X_te),
        "y_test": torch.from_numpy(y_te),
        "scaler": sc_y,
    }
    print(f"{t}: train={X_tr.shape}, test={X_te.shape}")

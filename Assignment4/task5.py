import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(1)
torch.manual_seed(1)

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

    # shuffle
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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"using {device}")

batch_size = 32
lr = 1e-3
weight_decay = 1e-5
epochs = 100
patience = 15
dropout = 0.20


class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop(out)
        out, _ = self.lstm2(out)
        return self.fc(out[:, -1, :])


def train_and_evaluate(ticker, data):
    X_tr = data["X_train"].to(device)
    y_tr = data["y_train"].to(device)
    X_te = data["X_test"].to(device)
    y_te = data["y_test"].to(device)
    sc = data["scaler"]

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    model = LSTMPredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []
    best_state, best_loss = None, float("inf")
    wait = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(xb)
        tr_loss = running / len(X_tr)

       
        model.eval()
        with torch.no_grad():
            te_loss = loss_fn(model(X_te), y_te).item()

        sched.step(te_loss)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)

        print(f"  epoch {ep:3d}  train={tr_loss:.6f}  test={te_loss:.6f}")

        if te_loss < best_loss:
            best_loss = te_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {ep}")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        tr_preds = model(X_tr).cpu().numpy()
        te_preds = model(X_te).cpu().numpy()
        tr_mse = loss_fn(model(X_tr), y_tr).item()
        te_mse = loss_fn(model(X_te), y_te).item()

    y_tr_real = sc.inverse_transform(y_tr.cpu().numpy()).flatten()
    y_tr_hat = sc.inverse_transform(tr_preds).flatten()
    y_te_real = sc.inverse_transform(y_te.cpu().numpy()).flatten()
    y_te_hat = sc.inverse_transform(te_preds).flatten()

    tr_rmse = np.sqrt(np.mean((y_tr_real - y_tr_hat) ** 2))
    te_rmse = np.sqrt(np.mean((y_te_real - y_te_hat) ** 2))
    tr_mape = np.mean(np.abs((y_tr_real - y_tr_hat) / y_tr_real)) * 100
    te_mape = np.mean(np.abs((y_te_real - y_te_hat) / y_te_real)) * 100
    acc = 100.0 * (np.abs(y_te_hat - y_te_real) / y_te_real <= 0.05).mean()

    print(f"\n  {ticker} done in {elapsed:.1f}s")
    print(f"  train  MSE={tr_mse:.6f}  RMSE=${tr_rmse:.2f}  MAPE={tr_mape:.2f}%")
    print(f"  test   MSE={te_mse:.6f}  RMSE=${te_rmse:.2f}  MAPE={te_mape:.2f}%  acc={acc:.1f}%\n")

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "y_test_true": y_te_real,
        "y_test_pred": y_te_hat,
        "train_mse": tr_mse,
        "test_mse": te_mse,
        "train_rmse": tr_rmse,
        "test_rmse": te_rmse,
        "train_mape": tr_mape,
        "test_mape": te_mape,
        "accuracy": acc,
        "elapsed": elapsed,
    }


# run
results = {}
for t in tickers:
    print(f"\n--- Training LSTM for {t} ---")
    results[t] = train_and_evaluate(t, datasets[t])



#plot
fig, axes = plt.subplots(4, 2, figsize=(14, 18))
fig.suptitle("Task 5 - LSTM", fontsize=14)

for i, t in enumerate(tickers):
    r = results[t]

    axes[i, 0].plot(r["train_losses"], label="train")
    axes[i, 0].plot(r["test_losses"], label="test")
    axes[i, 0].set_title(f"{t} – loss (scaled MSE)")
    axes[i, 0].set_xlabel("epoch")
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(r["y_test_true"], label="actual")
    axes[i, 1].plot(r["y_test_pred"], label="predicted", linestyle="--")
    axes[i, 1].set_title(f"{t} – MAPE={r['test_mape']:.2f}%  acc={r['accuracy']:.1f}%")
    axes[i, 1].set_xlabel("sample")
    axes[i, 1].set_ylabel("price ($)")
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task5_results.png", dpi=150)
print("saved task5_results.png")

print(f"\n{'Ticker':<6} {'Time':>6} {'Tr MSE':>10} {'Tr RMSE':>9} {'Tr MAPE':>9} {'Te MSE':>10} {'Te RMSE':>9} {'Te MAPE':>9} {'Acc':>7}")

for t in tickers:
    r = results[t]
    print(
        f"{t:<6} {r['elapsed']:>5.1f}s {r['train_mse']:>10.6f} {r['train_rmse']:>8.2f}$ {r['train_mape']:>8.2f}%"
        f" {r['test_mse']:>10.6f} {r['test_rmse']:>8.2f}$ {r['test_mape']:>8.2f}% {r['accuracy']:>6.1f}%"
    )
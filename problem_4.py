<<<<<<< HEAD
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from adaline_gd import AdalineGD
from adaline_sgd import AdalineSGD
from adaline_min_batch import AdalineMinibatch

# Load data
data = load_wine()
X = data.data
y = data.target

# Convert to binary for Adaline
y = (y == 0).astype(int)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Same hyperparameters
eta = 0.01
epochs = 50
batch_size = 32

# Train GD
gd = AdalineGD(eta=eta, n_iter=epochs)
start = time.time()
gd.fit(X, y)
time_gd = time.time() - start

# Train SGD
sgd = AdalineSGD(eta=eta, n_iter=epochs)
start = time.time()
sgd.fit(X, y)
time_sgd = time.time() - start

# Train Mini-batch
mini = AdalineMinibatch(eta=eta, n_iter=epochs, batch_size=batch_size)
start = time.time()
mini.fit(X, y)
time_mini = time.time() - start

# Report times
print("Training Time:")
print(f"GD: {time_gd:.4f} sec")
print(f"SGD: {time_sgd:.4f} sec")
print(f"Mini-batch: {time_mini:.4f} sec")

# Plot loss curves
plt.figure(figsize=(8,5))
plt.plot(gd.losses_, label="GD")
plt.plot(sgd.losses_, label="SGD")
plt.plot(mini.losses_, label="Mini-batch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Convergence Comparison")
plt.legend()
plt.grid(True)
=======
from models import LogisticRegressionGD as lr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# --- Wine Dataset ---
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                       'ml/machine-learning-databases/'
                       'wine/wine.data', header=None)

X_wine = df_wine.iloc[:, 1:].values
y_wine = df_wine.iloc[:, 0].values

# Class 1 and class 2
mask = (y_wine == 1) | (y_wine == 2)
X_wine = X_wine[mask]
y_wine = y_wine[mask]
y_wine = np.where(y_wine == 1, 0, 1)

# --- Hyperparameters (same for all three) ---
eta = 0.000001
n_iter = 1500
batch_size = 32

# --- GD ---
lr_gd = lr(eta=eta, n_iter=n_iter)
start = time.time()
lr_gd.fit(X_wine, y_wine)
time_gd = time.time() - start

# --- SGD ---
lr_sgd = lr(eta=eta, n_iter=n_iter)
start = time.time()
lr_sgd.fit_sgd(X_wine, y_wine)
time_sgd = time.time() - start

# --- Mini-Batch SGD ---
lr_mb = lr(eta=eta, n_iter=n_iter)
start = time.time()
lr_mb.fit_mini_batch_sgd(X_wine, y_wine, batch_size=batch_size)
time_mb = time.time() - start

print(f"GD      time: {time_gd:.4f}s, final loss: {lr_gd.losses_[-1]:.6f}")
print(f"SGD     time: {time_sgd:.4f}s, final loss: {lr_sgd.losses_[-1]:.6f}")
print(f"MB-SGD  time: {time_mb:.4f}s, final loss: {lr_mb.losses_[-1]:.6f}")

# --- Plot 1: Loss Convergence ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, n_iter + 1)
axes[0].plot(epochs, lr_gd.losses_, label='GD')
axes[0].plot(epochs, lr_sgd.losses_, label='SGD')
axes[0].plot(epochs, lr_mb.losses_, label=f'Mini-Batch SGD (batch={batch_size})')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Loss Convergence: GD vs SGD vs Mini-Batch SGD (Wine)')
axes[0].legend()

# --- Plot 2: Time Cost ---
methods = ['GD', 'SGD', f'Mini-Batch\nSGD (b={batch_size})']
times = [time_gd, time_sgd, time_mb]
bars = axes[1].bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Training Time Comparison (Wine)')
for bar, t in zip(bars, times):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{t:.3f}s', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("task4_plot.eps", format='eps')
plt.savefig("task4_plot.png", dpi=150)
>>>>>>> 7445970c5ceb75a49b3f0268da58a51afec3d70d
plt.show()
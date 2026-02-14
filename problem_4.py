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
plt.show()
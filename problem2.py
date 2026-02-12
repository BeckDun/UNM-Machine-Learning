from models import AdalineGD as ad
from models import LogisticRegressionGD as lr
from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
  
# --- Iris Dataset ---
iris = fetch_ucirepo(id=53)
X_iris = iris.data.features.values
y_iris = iris.data.targets.values.flatten()

# Filter for setosa (0) and versicolor (1) — first 100 samples
mask_iris = np.isin(y_iris, ['Iris-setosa', 'Iris-versicolor'])
X_iris = X_iris[mask_iris]
y_iris = y_iris[mask_iris]
# Encode labels: setosa -> 0, versicolor -> 1
y_iris = np.where(y_iris == 'Iris-setosa', 0, 1)

# --- Wine Dataset ---
wine = fetch_ucirepo(id=109)
X_wine = wine.data.features.values
y_wine = wine.data.targets.values.flatten()

# Filter for class 1 and class 2
mask_wine = np.isin(y_wine, [1, 2])
X_wine = X_wine[mask_wine]
y_wine = y_wine[mask_wine]
# Encode labels: 1 -> 0, 2 -> 1
y_wine = np.where(y_wine == 1, 0, 1)

# --- Standardize features ---
def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

X_iris_std = standardize(X_iris.astype(float))
X_wine_std = standardize(X_wine.astype(float))

# --- Hyperparameters (same for both models) ---
eta = 0.01
n_iter = 100

# --- Train on Iris ---
ada_iris = ad(eta=eta, n_iter=n_iter)
ada_iris.fit(X_iris_std, y_iris)

lr_iris = lr(eta=eta, n_iter=n_iter)
lr_iris.fit(X_iris_std, y_iris)

# --- Train on Wine ---
ada_wine = ad(eta=eta, n_iter=n_iter)
ada_wine.fit(X_wine_std, y_wine)

lr_wine = lr(eta=eta, n_iter=n_iter)
lr_wine.fit(X_wine_std, y_wine)

# --- Plot Loss Convergence ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Iris
axes[0].plot(range(1, n_iter + 1), ada_iris.losses_, label='Adaline')
axes[0].plot(range(1, n_iter + 1), lr_iris.losses_, label='Logistic Regression')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_title('Iris Dataset - Loss Convergence')
axes[0].legend()

# Wine
axes[1].plot(range(1, n_iter + 1), ada_wine.losses_, label='Adaline')
axes[1].plot(range(1, n_iter + 1), lr_wine.losses_, label='Logistic Regression')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].set_title('Wine Dataset - Loss Convergence')
axes[1].legend()

plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=150)
plt.show()
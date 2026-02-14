from models import AdalineGD as ad
from models import LogisticRegressionGD as lr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Iris Dataset ---
df_iris = pd.read_csv('https://archive.ics.uci.edu/'
                       'ml/machine-learning-databases/'
                       'iris/iris.data', header=None)

X_iris = df_iris.iloc[:, :4].values   
y_iris = df_iris.iloc[:, 4].values    

X_iris = X_iris[:100]
y_iris = y_iris[:100]
# Convert labels to 0 and 1
y_iris = np.where(y_iris == 'Iris-setosa', 0, 1)

# --- Wine Dataset (from textbook page 117-118) ---
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                       'ml/machine-learning-databases/'
                       'wine/wine.data', header=None)

X_wine = df_wine.iloc[:, 1:].values
y_wine = df_wine.iloc[:, 0].values

#class 1 and class 2
mask = (y_wine == 1) | (y_wine == 2)
X_wine = X_wine[mask]
y_wine = y_wine[mask]
# Convert labels to 0 and 1
y_wine = np.where(y_wine == 1, 0, 1)

# ---  hyperparameters 
eta_wine = 0.000001
eta_iris = 0.001
n_iter = 1500

# --- Train on Iris ---
ada_iris = ad(eta=eta_iris, n_iter=n_iter)
ada_iris.fit(X_iris, y_iris)

lr_iris = lr(eta=eta_iris, n_iter=n_iter)
lr_iris.fit(X_iris, y_iris)

# --- Train on Wine ---
ada_wine = ad(eta=eta_wine, n_iter=n_iter)
ada_wine.fit(X_wine, y_wine)

lr_wine = lr(eta=eta_wine, n_iter=n_iter)
lr_wine.fit(X_wine, y_wine)

# --- Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0][0].plot(range(1, n_iter + 1), ada_iris.losses_)
axes[0][0].set_xlabel('Epochs')
axes[0][0].set_ylabel('Mean Squared Error Loss')
axes[0][0].set_title('Iris Dataset - Adaline')

axes[0][1].plot(range(1, n_iter + 1), lr_iris.losses_)
axes[0][1].set_xlabel('Epochs')
axes[0][1].set_ylabel('Cross-Entropy Loss')
axes[0][1].set_title('Iris Dataset - Logistic Regression')

axes[1][0].plot(range(1, n_iter + 1), ada_wine.losses_)
axes[1][0].set_xlabel('Epochs')
axes[1][0].set_ylabel('Mean Squared Error Loss')
axes[1][0].set_title('Wine Dataset - Adaline')

axes[1][1].plot(range(1, n_iter + 1), lr_wine.losses_)
axes[1][1].set_xlabel('Epochs')
axes[1][1].set_ylabel('Cross-Entropy Loss')
axes[1][1].set_title('Wine Dataset - Logistic Regression')

plt.tight_layout()
plt.savefig("plot.eps")
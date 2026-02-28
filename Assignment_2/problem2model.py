from Assignment_2.problem2 import make_classification
import numpy as np
import matplotlib.pyplot as plt
X_training, X_test, y_training, y_test,a = make_classification(d=2,n=100,u=10,random_seed=1)

X = np.vstack((X_training,X_test))
y = np.hstack((y_training,y_test))

X_positive = X[y==1]
X_negative = X[y==-1]

plt.scatter(X_positive[:,0], X_positive[:,1], label='+1', s=20,color='red',marker='o')
plt.scatter(X_negative[:,0], X_negative[:,1], label='-1', s = 20, color='blue', marker='o')

x_values = np.linspace(X[:,0].min(),X[:,0].max(),100)
y_values = -(a[0]/a[1]) * x_values
plt.plot(x_values, y_values, label="Hyperplane", color='black')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Linearly Separable Data (d=2, n=100)")
plt.legend()
plt.grid(True)
plt.show()
import os
import pandas as pd
import numpy as np
from one_vs_rest_perceptron import OneVsRestPerceptron

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
dataset = pd.read_csv(s,header=None,encoding='utf-8')

X = dataset.iloc[:,0:4].values
y_strings = dataset.iloc[:,4]

label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
y = np.array([label_map[label] for label in y_strings])
model = OneVsRestPerceptron()
model.fit(X, y)

predictions = model.predict(X)
print(predictions)
print(y)


import numpy as np
from perceptron import Perceptron

class OneVsRestPerceptron:
    def __init__(self,eta=0.1,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.classifiers = {}
    
    def fit(self,X,y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y==cls, 1,0)
            classifier = Perceptron(self.eta,self.n_iter)
            classifier.fit(X,y_binary)
            self.classifiers[cls] = classifier
        return self
    

    def predict(self,X):
        scores = []
        for cls in self.classes_:
            clasifier = self.classifiers[cls]
            scores.append(clasifier.net_input(X))
        return self.classes_[np.argmax(scores, axis=0)]
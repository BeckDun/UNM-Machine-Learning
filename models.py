import numpy as np


class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        X_b = self._add_bias(X)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_b.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = X_b.dot(self.w_)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X_b.T.dot(errors) / X.shape[0]
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        X_b = self._add_bias(X)
        return X_b.dot(self.w_)

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class LogisticRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        X_b = self._add_bias(X)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_b.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = X_b.dot(self.w_)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X_b.T.dot(errors) / X.shape[0]
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        X_b = self._add_bias(X)
        return X_b.dot(self.w_)

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


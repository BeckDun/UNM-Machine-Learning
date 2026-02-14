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
    
    def fit_sgd(self, X, y):
        X_b = self._add_bias(X)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_b.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            indices = rgen.permutation(X_b.shape[0])
            X_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for xi, yi in zip(X_shuffled, y_shuffled):
                xi = xi.reshape(1, -1)
                net_input = xi.dot(self.w_)
                output = self.activation(net_input)
                error = yi - output
                self.w_ += self.eta * 2.0 * xi.T.dot(error)

            # Compute loss on data
            net_input_all = X_b.dot(self.w_)
            output_all = self.activation(net_input_all)
            loss = (-y.dot(np.log(output_all)) - (1 - y).dot(np.log(1 - output_all))) / X.shape[0]
            self.losses_.append(loss)
        return self
    
    def fit_mini_batch_sgd(self, X, y, batch_size=32):
       
        X_b = self._add_bias(X)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X_b.shape[1])
        self.losses_ = []
        n_samples = X_b.shape[0]

        for i in range(self.n_iter):
            indices = rgen.permutation(n_samples)
            X_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                net_input = X_batch.dot(self.w_)
                output = self.activation(net_input)
                errors = y_batch - output
                self.w_ += self.eta * 2.0 * X_batch.T.dot(errors) / X_batch.shape[0]

            # Compute loss 
            net_input_all = X_b.dot(self.w_)
            output_all = self.activation(net_input_all)
            loss = (-y.dot(np.log(output_all)) - (1 - y).dot(np.log(1 - output_all))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        X_b = self._add_bias(X)
        return X_b.dot(self.w_)

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


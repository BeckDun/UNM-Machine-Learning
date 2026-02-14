import numpy as np

class AdalineMinibatch:
    """ADAptive LInear NEuron classifier using Mini-batch SGD."""

    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None,
                 batch_size=32):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y):
        """Fit training data."""
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):

            if self.shuffle:
                X, y = self._shuffle(X, y)

            losses = []

            for start_idx in range(0, len(y), self.batch_size):
                X_batch = X[start_idx:start_idx + self.batch_size]
                y_batch = y[start_idx:start_idx + self.batch_size]

                losses.append(
                    self._update_weights_batch(X_batch, y_batch)
                )

            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def _update_weights_batch(self, X_batch, y_batch):
        """Apply Adaline learning rule for a mini-batch"""
        output = self.activation(self.net_input(X_batch))
        errors = (y_batch - output)

        self.w_ += self.eta * 2.0 * X_batch.T.dot(errors) / X_batch.shape[0]
        self.b_ += self.eta * 2.0 * errors.mean()

        loss = (errors ** 2).mean()
        return loss

    def partial_fit(self, X, y):
        """Fit without reinitializing weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Original SGD update"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        return error**2

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
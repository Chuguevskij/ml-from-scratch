import pandas as pd
import numpy as np
import random


class MySVM:
    def __init__(
        self, n_iter=10, learning_rate=0.001, C=1.0,
        sgd_sample=None, random_state=42
    ):
        # learning parameters
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.feature_names = None

        # model parameters
        self.weights = None
        self.bias = None

    def hinge_loss(self, X, y):
        """Hinge loss function."""
        margin_error = np.sum(np.square(self.weights))
        hinge_error = 1 / X.shape[0] * np.sum(
            np.maximum(0, 1 - y * (np.dot(X, self.weights) + self.bias))
        )
        return margin_error + hinge_error

    def _code_sign(self, y):
        """Prepare target for training."""
        return np.where(y == 0, -1, 1)

    def _decode_sign(self, y):
        """Decode target for prediction."""
        return np.where(y < 0, 0, 1)

    def _sgd_sample(self, X, y):
        """Sample dataset for stochastic gradient descent."""
        if isinstance(self.sgd_sample, int):
            rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
        elif isinstance(self.sgd_sample, float) and 0 < self.sgd_sample < 1:
            rows_idx = random.sample(
                range(X.shape[0]), round(X.shape[0] * self.sgd_sample)
            )
        else:
            raise ValueError(
                f"Invalid SGD sample type: {type(self.sgd_sample)}"
            )
        return X[rows_idx, :], y[rows_idx]

    def _fit_dataset(self, X, y):
        """
        Iterate over training dataset to
        update weights and bias.
        """
        for i in range(X.shape[0]):
            point_orientation = (
                y[i] * (np.dot(X[i, :], self.weights) + self.bias)
            )
            if point_orientation >= 1:
                # Right class, maximixe only margin
                delta_loss_weights = 2 * self.weights
                delta_loss_bias = 0
            else:
                # Wrong classification, minimize also loss
                delta_loss_weights = 2 * self.weights - self.C * y[i] * X[i, :]
                delta_loss_bias = -self.C * y[i]

            # Antigradient step
            self.weights -= self.learning_rate * delta_loss_weights
            self.bias -= self.learning_rate * delta_loss_bias

    def fit(self, X, y, verbose=False):
        """Fit SVM."""
        random.seed(self.random_state)

        # data preparation
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        y = np.array(y)
        y = self._code_sign(y)

        # if takes all sample for training
        X_sampled, y_sampled = X, y

        # learning parameters
        self.weights = np.ones(X.shape[1])
        self.bias = 1

        for n in range(1, self.n_iter+1):
            if self.sgd_sample:
                X_sampled, y_sampled = self._sgd_sample(X, y)
            self._fit_dataset(X_sampled, y_sampled)
            loss = self.hinge_loss(X, y)

            if verbose:
                # for start iter
                if n == 1:
                    print(f'start | loss: {loss}')
                # for every n iter
                if n % verbose == 0:
                    print(f'{n} | loss: {loss}')

    def predict(self, X):
        """Predict classes."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._decode_sign(y_pred)
        return y_pred

    def get_coef(self):
        return (self.weights.tolist(), self.bias)

    def __str__(self):
        return (f"{__class__.__name__} class: "
                f"n_iter={self.n_iter}, learning_rate={self.learning_rate}")

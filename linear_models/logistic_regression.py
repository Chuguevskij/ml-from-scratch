import pandas as pd
import numpy as np
import random

class MyLogReg:
    def __init__(self, n_iter=50, learning_rate=0.1,
                 eps=1e-15, weights=None):
        self._n_iter  = n_iter
        self._learning_rate = learning_rate
        self.weights = weights
        self.eps = eps

    def add_intercept(self, X):
        """Add intercept to feature array."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate([intercept, X], axis=1)

    def sigmoid(self, z):
        """Get logits."""
        return 1 / (1 + np.exp(-z))
    
    def gradient(self, X, y, y_hat):
        """Get a vector with partirial derivatives."""
        return (y_hat - y) @ X / X.shape[0]

    def cost_function(self, y, y_hat):
        """Get loss."""
        return -np.mean(
            y * np.log(y_hat + self.eps) + (1 - y) * np.log(1 - y_hat + self.eps)
        )

    def fit(self, X, y, verbose=False):
        """Train the model"""
        X = self.add_intercept(X)
        self.weights = np.ones(X.shape[1])

        for i in range(1, self._n_iter+1):
            z = X @ self.weights
            y_hat = self.sigmoid(z)
            loss = self.cost_function(y, y_hat)
            dw = self.gradient(X, y, y_hat)
            self.weights -= self._learning_rate * dw
            # display information
            if verbose:
                # for start iter
                if i == 1:
                    print(f'start | loss: {loss}')
                # for every i iter
                if i % verbose == 0:
                    print(f'{i} | loss: {loss}')

    def __str__(self):
        return f"{__class__.__name__} class: n_iter={self._n_iter}, learning_rate={self._learning_rate}"

    def get_coef(self):
        return self.weights[1:]


df_desease = (
    pd.read_csv('C:/Users/chugu/datasets/classification/alzheimers_disease/alzheimers_disease_data.csv')
    .drop(columns=['DoctorInCharge', 'PatientID']))
X = df_desease.loc[:, df_desease.columns != 'Diagnosis']
X = (X-X.min())/(X.max()-X.min())
y = df_desease['Diagnosis']
train_size = len(X) // 3 * 2  # 2/3 for train
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test, y_test = X.iloc[:train_size], y.iloc[:train_size]

log_reg = MyLogReg(n_iter=50, learning_rate=0.01)
log_reg.fit(X_train, y_train, verbose=25)
#print(log_reg.get_coef())
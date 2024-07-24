import pandas as pd
import numpy as np
import random

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self._n_iter  = n_iter
        self._learning_rate = learning_rate

    @staticmethod
    def _logloss(y, y_hat):
        return np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def _grad(obj_num, y, y_hat, weights):
        return 1 / obj_num * 

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        #self._y = y
        self._X = X
        self._X.insert(0, 'bias', 1)
        self._weights = np.ones(self._X.shape[1])

        for n in range(1, self._n_iter+1):
            # predict
            y_hat = 1 / (1 + np.exp(self._X @ self._weights))
            # evaluate loss
            self.loss = self._logloss(y, y_hat)
            # gradient step
            partial_w = 


    def __str__(self):
        return f"{__class__.__name__} class: n_iter={self._n_iter}, learning_rate={self._learning_rate}"
    
log_reg = MyLogReg()

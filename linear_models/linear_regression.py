import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None,
                 reg=None, l1_coef=0, l2_coef=0):
        self._n_iter  = n_iter
        self._learning_rate = learning_rate
        self._metric = metric
        self._reg = reg
        self._l1_coef = l1_coef
        self._l2_coef = l2_coef

    @staticmethod
    def _mean_squared_error(y, y_hat):
        '''
        Private method, used to evaluate loss at each iteration.

        :param: y - array, true values
        :param: y_hat - array, predicted values
        :return: float
        '''
        return np.mean((y - y_hat) ** 2)
    
    @staticmethod
    def _mae(y, y_hat):
        return np.mean(np.abs(y - y_hat))
    
    @staticmethod
    def _mse(y, y_hat):
        return np.mean((y - y_hat) ** 2)
    
    @staticmethod
    def _rmse(y, y_hat):
        return np.sqrt(np.mean((y - y_hat) ** 2))
    
    @staticmethod
    def _mape(y, y_hat):
        return np.mean(np.abs((y - y_hat )/ y)) * 100
    
    @staticmethod
    def _r2(y, y_hat):
        ssr = np.sum((y - y_hat) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        return 1 - (ssr / sst)
    
    def _l1(self):
        loss_penalty = self._l1_coef * np.abs(self._weights).sum()
        grad_penalty = self._l1_coef * self._weights / np.abs(self._weights)
        return loss_penalty, grad_penalty
    
    def _l2(self):
        loss_penalty = self._l2_coef * np.square(self._weights).sum()
        grad_penalty = self._l2_coef * 2 * self._weights
        return loss_penalty, grad_penalty
    
    def _elasticnet(self):
        return self._l1() + self._l2()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        self._X = X
        self._X.insert(0, 'bias', 1)
        self._weights = np.ones(self._X.shape[1]) 

        for n in range(1, self._n_iter+1):
            # predict
            y_hat = self._X @ self._weights
            # evaluate loss
            penalty = getattr(self, '_' + self._reg)() if self._reg else (0, 0)
            loss = self._mean_squared_error(y, y_hat) + penalty[0]
            # gradient step
            partial_w = 2 / self._X.shape[0] * (y_hat - y) @ self._X + penalty[1]
            # choose learning rate
            if hasattr(self._learning_rate, '__call__'):
                lr = self._learning_rate(n)
            else:
                lr = self._learning_rate
            self._weights -= lr * partial_w
            lr_score = f'| learning rate {lr}'

            # get metric
            if self._metric:            
                score = getattr(self, '_' + self._metric)(y, y_hat)
                verbose_score = f' | {self._metric}: {score}'
            else:
                verbose_score = ''
            
            # display information
            if verbose:
                # for start iter
                if n == 1:
                    print(f'start | loss: {loss}' + verbose_score + lr_score)
                # for every n iter
                if n % verbose == 0:
                    print(f'{n} | loss: {loss}' + verbose_score + lr_score)

    def get_best_score(self):
        if self._metric:
            return getattr(self, '_' + self._metric)(y, self._X @ self._weights)
        else:
            return "Metric is not defined!"
    
    def predict(self, X: pd.DataFrame):
        X = X.copy(deep=True)
        X.insert(loc=0, column='bias', value=1)
        return np.dot(X, self._weights)

    def get_coef(self):
        return self._weights[1:]
    
    def __str__(self):
        return f"{__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
df_prices = pd.read_csv('C:/Users/chugu/datasets/linear_reg/boston_house_prices.csv')
X = df_prices.loc[:, df_prices.columns != 'MEDV']
X = (X-X.min())/(X.max()-X.min())
y = df_prices['MEDV']
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[:400], y[:400]

lin_reg = MyLineReg(n_iter=250, learning_rate=lambda iter: 0.95 ** iter, reg='elasticnet', l1_coef=0.1, l2_coef=0.1)
lin_reg.fit(X_train, y_train, verbose=50)
print(lin_reg.get_best_score())
y_pred = lin_reg.predict(X_test)
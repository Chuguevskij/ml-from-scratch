import pandas as pd
import numpy as np
import random

class MyLogReg:
    def __init__(self, n_iter=50, learning_rate=0.1, metric=None,
                 eps=1e-15, weights=None):
        self._n_iter  = n_iter
        self._learning_rate = learning_rate
        self._metric = metric
        self.eps = eps
        self._weights = weights
        self._X = None
        self.score = None

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
        self._X = self.add_intercept(X)
        self._weights = np.ones(self._X.shape[1])

        for i in range(1, self._n_iter+1):
            z = self._X @ self._weights
            y_hat = self.sigmoid(z)
            loss = self.cost_function(y, y_hat)
            dw = self.gradient(self._X, y, y_hat)
            self._weights -= self._learning_rate * dw

            # get metric
            if self._metric:            
                self.score = getattr(self, '_' + self._metric)(y, y_hat)
                verbose_score = f' | {self._metric}: {self.score}'
            else:
                verbose_score = ''

            # display information
            if verbose:
                # for start iter
                if i == 1:
                    print(f'start | loss: {loss}' + verbose_score)
                # for every i iter
                if i % verbose == 0:
                    print(f'{i} | loss: {loss}' + verbose_score)

    def predict_proba(self, X):
        """Return probabilities of class."""
        X = self.add_intercept(X)
        return self.sigmoid(X @ self._weights)

    def predict(self, X, threshold=0.5):
        """Return classes."""
        return np.where(self.predict_proba(X) >= threshold, 1, 0)

    @staticmethod
    def get_class_stats(y, y_hat, threshhold = 0.5):
        """Get confusion matrix."""
        y_hat = np.where(y_hat >= threshhold, 1, 0)
        compare = np.stack([y, y_hat], axis=1)
        tp = len(compare[(compare[:, 0] == 1) & (compare[:, 1] == 1)])
        tn = len(compare[(compare[:, 0] == 0) & (compare[:, 1] == 0)])
        fp = len(compare[(compare[:, 0] == 0) & (compare[:, 1] == 1)])
        fn = len(compare[(compare[:, 0] == 1) & (compare[:, 1] == 0)])
        return tp, tn, fp, fn

    @staticmethod
    def _accuracy(y, y_hat):
        """Get accuracy."""
        tp, tn, fp, fn = __class__.get_class_stats(y, y_hat)
        return (tp + tn) / (tp + tn + fp + fn)

    @staticmethod
    def _precision(y, y_hat):
        """
        Compute share of realy true positive.
        """
        tp, tn, fp, fn = __class__.get_class_stats(y, y_hat)
        return tp / (tp + fp)

    @staticmethod
    def _recall(y, y_hat):
        """
        Shows ability to distinguish 
        positive class among all ones.
        """
        tp, tn, fp, fn = __class__.get_class_stats(y, y_hat)
        return tp / (tp + fn)

    @staticmethod
    def _f1(y, y_hat, beta=1):
        """Combination of recall and precision."""
        #tp, tn, fp, fn = __class__.get_class_stats(y, y_hat)
        recall = __class__._recall(y, y_hat)
        precision = __class__._precision(y, y_hat)
        return (
            (1 + beta**2) * (precision * recall)
            / (beta * precision + recall)
            )

    @staticmethod
    def _roc_axises(y, y_hat):
        """
        Get axises for plot:
        TPR - true positive rate (recall)
        FPR - false positive rate
        """
        pass

    @staticmethod
    def _roc_auc(y, y_hat):
        """
        Compute area under
        the receiver operating characteristics.
        """
        y_1 = np.array(y)
        y_2 = np.array(y_hat)
        compare = pd.DataFrame([y_1, y_2]).T
        compare = compare.sort_values(1, ascending=False).reset_index(drop=True)
        compare[1] = compare[1].round(10)
        compare['s_1'] = compare[0].rolling(len(compare), min_periods = 1).sum()
        compare.loc[compare[0] == 1, 's_1'] = 0
        compare['s_2'] = compare.groupby([1])[0].transform(lambda x: x.sum() / 2)
        compare.loc[compare[0] == 1, 's_2'] = 0
        t_num = compare[compare[0] == 1].shape[0]
        f_num = compare[compare[0] == 0].shape[0]
        return (compare['s_1'].sum() + compare['s_2'].sum()) / t_num / f_num

    def get_best_score(self):
        """Get metric for trained model."""
        return self.score

    def get_coef(self):
        """Get trained weights except the bias."""
        return self._weights[1:]

    def __str__(self):
        return f"{__class__.__name__} class: n_iter={self._n_iter}, learning_rate={self._learning_rate}"



df_desease = (
    pd.read_csv('C:/Users/chugu/datasets/classification/alzheimers_disease/alzheimers_disease_data.csv')
    .drop(columns=['DoctorInCharge', 'PatientID']))
X = df_desease.loc[:, df_desease.columns != 'Diagnosis']
X = (X-X.min())/(X.max()-X.min())
y = df_desease['Diagnosis']
train_size = len(X) // 3 * 2  # 2/3 for train
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
log_reg = MyLogReg(n_iter=50, learning_rate=0.1, metric='roc_auc')
log_reg.fit(X_train, y_train, verbose=10)

y_pred = log_reg.predict(X_test)
#print(log_reg._accuracy(y_test, y_pred))
print(log_reg.get_best_score())
#print(log_reg.get_coef())
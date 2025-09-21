import numpy as np


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))


def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y)) * 100


def r2(y, y_hat):
    ssr = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - (ssr / sst)

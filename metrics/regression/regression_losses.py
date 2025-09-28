import numpy as np


def neg_grad_mse(y, y_pred):
    """Negative gradient for MSE."""
    return y - y_pred


def neg_grad_mae(y, y_pred):
    """Negative gradient for MAE."""
    return np.sign(y - y_pred)

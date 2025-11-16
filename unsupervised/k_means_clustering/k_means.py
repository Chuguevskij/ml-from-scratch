import random

import numpy as np
import pandas as pd

from metrics.distance import *


class MyKMeans():
    def __init__(
        self,
        n_clusters=3,
        max_iter=10,
        n_init=3,
        random_state=42
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __str__(self):
        params = [
            "n_clusters",
            "max_iter",
            "n_init",
            "random_state"
        ]
        attrs = ", ".join(f"{p}={getattr(self, p)}" for p in params)
        return f"{self.__class__.__name__} class: {attrs}"

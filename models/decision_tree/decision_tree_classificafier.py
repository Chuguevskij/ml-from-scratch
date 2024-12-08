import pandas as pd
import numpy as np


class MyTreeClf():
    def __init__(self, max_depth = 5, min_samples_split = 2,
                 max_leafs = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        columns = X.columns.tolist()
        #splits = []

        for col_name in columns:
            splits = X[col_name].unique().sort_values().tolist()
        return splits


    def __str__(self):
        return f"{__class__.__name__} class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"



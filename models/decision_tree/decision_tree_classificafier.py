import numpy as np
import pandas as pd


class DecisionNode():
    def __init__(self, feature_i, threshold, left_branch, right_branch, value):
        self.feature_i = feature_i  # feature index to split by
        self.threshold = threshold  # threshold value to split by
        self.left_branch = left_branch  # left branch
        self.right_branch = right_branch  # right branch
        self.value = value  # value if the node is a leaf in the tree


class MyTreeClf():
    def __init__(
            self,
            max_depth=5,
            min_samples_split=2,
            max_leafs=20):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    @staticmethod
    def _shannon_entropy(y, eps=1e-12):
        """Calculate shannon_entropy for a single sample."""
        p_0 = len(y[y == 0]) / len(y)
        p_1 = len(y[y == 1]) / len(y)
        return -(
            p_0 * np.log2(p_0+eps) +
            p_1 * np.log2(p_1+eps)
        )

    @staticmethod
    def _information_gain(y, y_left, y_right):
        """Calculate information gain for a single split."""
        n = len(y)
        share_left = len(y_left) / n
        share_right = len(y_right) / n
        return (
            MyTreeClf._shannon_entropy(y) -
            share_left * MyTreeClf._shannon_entropy(y_left) -
            share_right * MyTreeClf._shannon_entropy(y_right)
        )

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        """Find the best feature and best value to split."""
        list_features = X.columns
        X = X.to_numpy()
        y = y.to_numpy()
        y = np.expand_dims(y, axis=1)
        Xy = np.concatenate((X, y), axis=1)
        num_features = X.shape[1]

        largest_ig = -100
        best_criteria = None

        # Calculate the impurity for each feature
        for feature_i in range(num_features):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            splits = np.unique(feature_values)
            splits = np.sort(splits)
            splits = [
                (splits[i + 1] + splits[i]) / 2 for i in range(len(splits) - 1)
                ]

            # Iterate through all unique values of feature column i and
            # calculate the impurity
            for threshold in splits:
                Xy_left = Xy[Xy[:, feature_i] <= threshold]
                Xy_right = Xy[Xy[:, feature_i] > threshold]

                if len(Xy_left) > 0 and len(Xy_right) > 0:
                    # Select the y-values of the two sets
                    y_left = Xy_left[:, num_features:]
                    y_right = Xy_right[:, num_features:]

                    # Calculate impurity
                    ig = MyTreeClf._information_gain(y, y_left, y_right)
                    if ig > largest_ig:
                        best_criteria = {
                            "feature": list_features[feature_i],
                            "threshold": threshold,
                            "information_gain": ig
                        }
                        largest_ig = ig
        return list(best_criteria.values())

    def __str__(self):
        return (
            f"{__class__.__name__} class: max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_leafs={self.max_leafs}"
            )

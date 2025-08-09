import numpy as np
from tree_node import DecisionNode


class MyTreeClf():
    """
    A simple implementation of a Decision Tree Classifier.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_leaves=20
    ):
        """
        Initializes a Decision Tree.

        Args:
            max_depth (int): Maximum depth of the decision tree. Defaults to 5.
            min_samples_split (int): Minimum number of samples required
            to split. Defaults to 2.
            max_leaves (int): Maximum number of leaves in the tree. Defaults
            to 20.
        """
        self.root = None
        self.leaves_cnt = 0
        self.potential_leaf_cnt = 0

        # Building restriction parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaves = max_leaves

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

    def _get_leaf_value(self, y, th=0.5):
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        # Most occurring value
        values, counts = np.unique(y, return_counts=True)
        most_occuring_value = values[np.argmax(counts)]

        # Probability of class 1
        prob_class_1 = np.mean(y == 1)

        return [prob_class_1, most_occuring_value]

    def _get_best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best feature and threshold to split."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_split = None

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)
            thresholds = np.sort(thresholds)
            if len(thresholds) <= 1:
                continue
            splits = [
                (thresholds[i] + thresholds[i + 1]) / 2
                for i in range(len(thresholds) - 1)
            ]

            for threshold in splits:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_i': feature_i,
                        'threshold': threshold,
                        'gain': gain,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        return best_split

    def check_stopping_criteria(self, y, depth):
        """Check the criteria to stop building the tree."""
        n_samples = len(y)
        n_labels = len(np.unique(y))
        too_many_leaves = (
            self.leaves_cnt + self.potential_leaf_cnt >= self.max_leaves
        )
        if (
            n_labels == 1 or
            depth >= self.max_depth or
            too_many_leaves or
            n_samples <= self.min_samples_split
        ):
            return True
        return False

    def _build_tree(self, X, y, depth=0, is_potential=False):
        # If it was counted as potential before, convert it to real node now
        if is_potential:
            self.potential_leaf_cnt = max(0, self.potential_leaf_cnt - 1)

        if self.check_stopping_criteria(y, depth):
            self.leaves_cnt += 1
            return DecisionNode(value=self._get_leaf_value(y))

        split = self._get_best_split(X, y)

        if split is None or split['gain'] <= 0 or split['feature_i'] is None:
            self.leaves_cnt += 1
            return DecisionNode(value=self._get_leaf_value(y))

        # Now we're going to split, so we add 2 new potential leaves
        self.potential_leaf_cnt += 2

        feature = split['feature_i']
        threshold = split['threshold']
        left_mask = split['left_mask']
        right_mask = split['right_mask']

        left = self._build_tree(
            X[left_mask], y[left_mask], depth + 1, is_potential=True)
        right = self._build_tree(
            X[right_mask], y[right_mask], depth + 1, is_potential=True)

        return DecisionNode(
            feature_i=feature,
            threshold=threshold,
            gain=split['gain'],
            left_branch=left,
            right_branch=right,
        )

    def fit(self, X, y):
        """Fit the decision tree classifier to the data."""
        X = np.array(X)
        y = np.array(y)
        self.leaves_cnt = 0
        self.root = self._build_tree(X, y)

    def predict_proba(self, X_pred):
        "Predict probability for the 1st class."
        if self.root is None:
            print('Tree is not build')
            return []

        X_pred = np.array(X_pred)

        def _predict_proba_sample(node, x):
            if node.value is not None:
                return node.value[0]
            if x[node.feature_i] <= node.threshold:
                return _predict_proba_sample(node.left_branch, x)
            else:
                return _predict_proba_sample(node.right_branch, x)

        return np.array([_predict_proba_sample(self.root, x) for x in X_pred])

    def predict(self, X_pred):
        "Predict class."
        probs = self.predict_proba(X_pred)
        return (probs > 0.5).astype(int)

    def bft(self):
        "Breadth-first traversal."
        if self.root is None:
            print('Tree is not build')
            return []

        queue = []
        res = []

        level = 0
        queue.append(self.root)

        while queue:
            res.append([])
            for _ in range(len(queue)):
                node = queue.pop(0)

                if node.value is None:
                    node_data = (node.feature_i, node.threshold)
                else:
                    node_data = (node.value)
                res[level].append(node_data)

                if node.left_branch is not None:
                    queue.append(node.left_branch)

                if node.right_branch is not None:
                    queue.append(node.right_branch)
            level += 1

        return res

    def print_tree(self):
        """
        Print of tree levels from bft().
        """
        levels = self.bft()
        for level_idx, nodes in enumerate(levels):
            indent = "  " * level_idx
            for node in nodes:
                if isinstance(node, tuple):
                    feature_i, threshold = node
                    print(
                        f"{indent}Level {level_idx}: "
                        f"(feature {feature_i} <= {threshold})"
                    )
                else:
                    print(f"{indent}Level {level_idx}: Predict: {node}")

    def __str__(self):
        return (
            f"{__class__.__name__} class: max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_leaves={self.max_leaves}"
            )

class DecisionNode:
    """
    Represents a node or leaf in a decision tree.
    """

    def __init__(
        self,
        feature_i=None,
        threshold=None,
        gain=None,
        left_branch=None,
        right_branch=None,
        value=None
    ):
        """
        Initializes a DecisionNode.

        Args:
            feature_i (int, optional): Feature index to split by.
            threshold (float, optional): Threshold value to split by.
            gain (float, optional): Information gain from the split.
            left_branch (DecisionNode, optional): Left branch of the tree.
            right_branch (DecisionNode, optional): Right branch of the tree.
            value (Any, optional): Value if the node is a leaf in the tree.
        """
        self.feature_i = feature_i
        self.threshold = threshold
        self.gain = gain
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.value = value

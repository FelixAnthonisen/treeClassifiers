import numpy as np
from typing import Self
"""
This is a suggested template and you do not need to follow it. You can change any part of it to fit your needs.
There are some helper functions that might be useful to implement first.
At the end there is some test code that you can use to test your implementation on synthetic data by running this file.
"""


def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2, 5])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    _, counts = np.unique(y, return_counts=True)

    proportions = counts / len(y)
    return proportions


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    return 1 - np.sum(count(y) ** 2)


def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    return -np.sum(count(y) * np.log2(count(y)))


def mask(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    return x <= value


def split(x: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray]:
    """
    Return a split based on a mask
    """
    return (x[msk], x[~msk])


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    counts = np.bincount(y)
    return np.argmax(counts)


def identical_feature_vals(x: np.ndarray) -> bool:
    """
    Return True if all the values in x are the same.
    Example:
        identical_feature_vals(np.array([1, 1, 1, 1, 1])) -> True
        identical_feature_vals(np.array([1, 2, 3, 4, 5])) -> False
    """
    return len(set(x)) == 1


def identical_feature_values(x: np.ndarray) -> bool:
    b = True
    for col in x.T:
        b = b and len(set(col)) == 1
    return b


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
        Return True if the node is a leaf node
        """
        return self.value is not None

    def pretty_print(self, level: int = 0, prefix: str = "") -> None:
        """
        Pretty print the decision tree starting from this node.
        """
        if self.is_leaf():
            print(f"{prefix}{' ' * (level * 4)}[Leaf] Value: {self.value}")
        else:
            print(f"""{prefix}{' ' * (level * 4)
                               }[Feature {self.feature}] <= {self.threshold}""")
            if self.left is not None:
                if self.right is not None:
                    # Print left and right branches
                    self.left.pretty_print(level + 1, prefix + "├── ")
                    self.right.pretty_print(level + 1, prefix + "└── ")
                else:
                    # Only print left branch
                    self.left.pretty_print(level + 1, prefix + "└── ")
            else:
                if self.right is not None:
                    # Only print right branch
                    self.right.pretty_print(level + 1, prefix + "└── ")


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
    ) -> None:
        self.root = None
        self.max_depth = max_depth
        if criterion == "entropy":
            self.info_func = entropy
        elif criterion == "gini":
            self.info_func = gini_index
        else:
            raise ValueError(
                "Invalid criterion: Must be either 'entropy' or 'gini'")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """
        if len(X) == 0:
            raise ValueError("Cannot fit empty dataset")

        if len(y) != len(X):
            raise ValueError("Values and targets must be same length")

        self.root = self.recurse(X, y, 0)

    def recurse(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:

        if len(set(y)) == 1:
            return Node(value=y[0])

        max_depth_reached = (self.max_depth and depth >= self.max_depth)

        if identical_feature_values(X) or max_depth_reached:
            return Node(value=most_common(y))

        N = len(y)

        max_info_gain = -float("inf")
        best_feature = None
        threshold = 0.0

        for i, features in enumerate(X.T):
            mean = np.mean(features)
            msk = mask(features, mean)
            left, right = split(X, msk)

            info_gain = (
                self.info_func(X) -
                (len(left)/N*self.info_func(left) +
                 len(right)*self.info_func(right))
            )

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = i
                threshold = mean

        msk = mask(X.T[best_feature], threshold)

        X_left, X_right = split(X, msk)
        y_left, y_right = split(y, msk)
        left_child = self.recurse(X_left, y_left, depth+1)
        right_child = self.recurse(X_right, y_right, depth+1)

        return Node(feature=best_feature, threshold=threshold, left=left_child, right=right_child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        if not self.root:
            raise ValueError("fit must be called before calling predict")
        return np.array([self.recurse2(self.root, x) for x in X])

    def recurse2(self, root: Node, x: np.ndarray) -> int:
        if root.is_leaf():
            return root.value
        feature_value = x[root.feature]
        if feature_value <= root.threshold:
            return self.recurse2(root.left, x)
        return self.recurse2(root.right, x)

    def score(self, y: np.ndarray, predicted_y) -> np.ndarray:
        N = len(y)
        return np.count_nonzero(predicted_y == y)/N


if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 123

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    for d in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        pingu = print
        pingu("Depth: ", d)
        rf = DecisionTree(criterion="gini", max_depth=d)
        rf.fit(X_train, y_train)
        print("Training accuracy: ", accuracy_score(
            y_train, rf.predict(X_train)))
        print("Validation accuracy: ", accuracy_score(y_val, rf.predict(X_val)))
        print(f"Custom score: {rf.score(y_val, rf.predict(X_val))}")
        print("--------------------------------------------------------------------------------------")

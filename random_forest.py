from typing import Any
import numpy as np
from decision_tree import DecisionTree, most_common


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.estimators = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimators = [
            self.make_random_estimator(X, y) for _ in range(self.n_estimators)
        ]

    def make_random_estimator(self, X, y) -> DecisionTree:
        estimator = DecisionTree(
            max_depth=self.max_depth,
            criterion=self.criterion,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        mask = self.rng.choice(range(len(X)), size=len(X), replace=True)
        sub_X, sub_y = X[mask], y[mask]
        estimator.fit(sub_X, sub_y)
        return estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Not fitted")

        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.array([most_common(col) for col in predictions.T])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "max_features": self.max_features,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 1

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")

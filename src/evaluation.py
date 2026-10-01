import numpy as np
from sklearn.metrics import make_scorer


class LabeledStratifiedKFold:
    def __init__(self, n_splits, *, shuffle=False, random_state=None):
        if not shuffle and random_state is not None:
            raise ValueError("random_state has no effect when shuffle=False.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups=None):
        positions = np.arange(len(X))
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            positions = rng.permutation(positions)

        y_values = np.asarray(y)[positions]
        positive_positions = positions[y_values == 1]
        negative_positions = positions[y_values == 0]

        positive_folds = np.array_split(positive_positions, self.n_splits)
        negative_folds = np.array_split(negative_positions, self.n_splits)

        for i in range(self.n_splits):
            test_positions = np.concatenate(
                [positive_folds[i], negative_folds[i]]
            )
            train_positions = np.setdiff1d(positions, test_positions)
            yield train_positions, test_positions

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def cost(y_true, y_pred_proba):
    # Exact minimum cost via closed-form optimal threshold search.
    y_true = np.asarray(y_true)
    p = y_pred_proba[:, 1]
    order = np.argsort(p)
    y_sorted = y_true[order]

    c1 = np.cumsum(y_sorted * 745 - 150)
    c2 = np.cumsum(y_sorted * 155 - 150)

    return c1.min() + c2.min() + (y_sorted == 0).sum() * 300


def get_scorer():
    return make_scorer(
        cost, response_method="predict_proba", greater_is_better=False
    )

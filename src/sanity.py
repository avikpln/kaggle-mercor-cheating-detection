# -- Imports --
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score

from data import load_train_data
from evaluation import LabeledStratifiedKFold, get_scorer

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of cross-validation folds.
N_SPLITS = 5

# -- Oracle --
class Oracle(BaseEstimator, ClassifierMixin):
    def __init__(self, y_true):
        self.y_true = y_true

    def fit(self, X, y):
        self.classes_ = np.unique(self.y_true)
        return self

    def predict_proba(self, X):
        y = self.y_true.loc[X.index].to_numpy()
        return np.column_stack([y == c for c in self.classes_])

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

# -- Testing --
def test_oracle():
    data_train = load_train_data().set_index("user_hash")
    data_train = data_train.sample(frac=1.0, random_state=RANDOM_STATE)
    target_name = "is_cheating"
    target_train = data_train[target_name]
    data_train = data_train.drop(columns=[target_name])
    data_train = data_train.drop(columns=["high_conf_clean"])

    model = Oracle(target_train)

    cv = LabeledStratifiedKFold(
        N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    scores = cross_val_score(
        model,
        data_train,
        target_train,
        cv=cv,
        scoring=get_scorer(),
        verbose=2,
        # n_jobs=-1,
    )
    print(scores)

# -- Main --
if __name__ == "__main__":
    test_oracle()
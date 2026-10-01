# -- Imports --
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.semi_supervised import SelfTrainingClassifier

from data import load_social_graph, load_train_data
from evaluation import LabeledStratifiedKFold

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of cross-validation folds.
N_SPLITS = 5

# -- Data --
def _load_data(transformers=()):
    data_train = load_train_data().set_index("user_hash")
    data_train = data_train.drop(columns=["high_conf_clean"])
    target_name = "is_cheating"
    y = data_train[target_name]
    X = data_train.drop(columns=[target_name])
    for transformer in transformers:
        X, y = transformer(X, y)
    return X, y

# -- Data Transformations --
def transform_remove_graph_samples(X, y):
    graph = load_social_graph()
    graph_users = set(graph["user_a"]) | set(graph["user_b"])
    outside = ~X.index.isin(graph_users)
    return X[outside], y[outside]


def transform_remove_unlabeled(X, y):
    labeled = y.notna()
    return X[labeled], y[labeled]


def transform_nan_labels_to_neg1(X, y):
    return X, y.fillna(-1)

# -- Factory --
def get_hgbdt():
    return HistGradientBoostingClassifier(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

# -- Testing --
def evaluate(X, y, classifier, note):
    cv = LabeledStratifiedKFold(
        N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_validate(
        classifier,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        verbose=2,
    )
    test_scores = scores["test_score"]
    print(f"[{note}]")
    print(f"Scores: {test_scores}")
    print(f"Score std: {np.std(test_scores):.4f}")
    print(f"Mean score: {np.mean(test_scores):.4f}")


def test_hgbdt():
    transformers = (
        transform_remove_graph_samples,
        transform_remove_unlabeled,
    )
    X, y = _load_data(transformers)
    classifier = get_hgbdt()
    evaluate(X, y, classifier, "HGBDT")


def test_ss_hgbdt():
    transformers = (
        transform_remove_graph_samples,
        transform_nan_labels_to_neg1,
    )
    X, y = _load_data(transformers)
    classifier = SelfTrainingClassifier(get_hgbdt())
    evaluate(X, y, classifier, "SS HGBDT")

# -- Main --
if __name__ == "__main__":
    # test_hgbdt()
    test_ss_hgbdt()

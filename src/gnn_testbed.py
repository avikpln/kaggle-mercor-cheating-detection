# -- Imports --
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from data import load_social_graph, load_train_data
from evaluation import LabeledStratifiedKFold
from imputation import get_imputer
from gnn_one_hop import OneHopGNNClassifier
from gnn_conv import ConvGNNClassifier

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of cross-validation folds.
N_SPLITS = 5

# -- Data --
def _load_data():
    data_train = load_train_data().set_index("user_hash")
    graph = load_social_graph()
    graph_users = set(graph["user_a"]) | set(graph["user_b"])
    data_train = data_train.loc[data_train.index.intersection(graph_users)]
    data_train = data_train.drop(columns=["high_conf_clean"])
    target_name = "is_cheating"
    y = data_train[target_name]
    X = data_train.drop(columns=[target_name])
    return X, y

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

def test_one_hop_gnn():
    X, y = _load_data()
    classifier = make_pipeline(
        get_imputer(random_state=RANDOM_STATE),
        OneHopGNNClassifier(random_state=RANDOM_STATE),
    )
    evaluate(
        X,
        y,
        classifier,
        "One-hop GNN",
    )

def test_conv_gnn():
    X, y = _load_data()
    classifier = make_pipeline(
        get_imputer(random_state=RANDOM_STATE),
        ConvGNNClassifier(random_state=RANDOM_STATE),
    )
    evaluate(
        X,
        y,
        classifier,
        "Conv GNN",
    )

# -- Main --
if __name__ == "__main__":
    # test_one_hop_gnn()
    test_conv_gnn()

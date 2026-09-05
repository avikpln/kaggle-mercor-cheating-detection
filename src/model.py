# -- Import Libraries --
import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data import load_train_data, load_social_graph
from imputation import get_imputer
from pseudo_labeling import get_labeler
from semi_supervised import SemiSupervisedClassifier

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of cross-validation folds.
N_SPLITS = 5

# -- Data Preparation --

# Load the training data from a CSV file.
data_train = load_train_data().set_index("user_hash")

# Shuffle the training data.
data_train = data_train.sample(frac=1.0, random_state=RANDOM_STATE)

# Separate the target variable from the training data.
target_name = "is_cheating"
target_train = data_train[target_name]
data_train = data_train.drop(columns=[target_name])

# Feature selection.
data_train = data_train.drop(columns=["feature_014", "high_conf_clean"])

# Load social graph data from a CSV file.
social_graph = load_social_graph()

# -- Feature Imputation --
imputer = get_imputer(random_state=RANDOM_STATE)

# -- Preprocessing --
preprocessor = FunctionTransformer()

# -- Pseudo-Labeling --
graph = nx.from_pandas_edgelist(
    social_graph,
    source="user_a",
    target="user_b",
)

labeler = get_labeler(graph, random_state=RANDOM_STATE)

# -- Classification --
classifier = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE,
    max_iter=100,
    learning_rate=0.1,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
)

estimator = SemiSupervisedClassifier(classifier, labeler)

# -- Pipeline --
pipeline = Pipeline([
    ("imputer", imputer),
    ("preprocessor", preprocessor),
    ("estimator", estimator),
])

# -- Training and Evaluation --
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

cv=LabeledStratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scorer = make_scorer(
    cost, response_method="predict_proba", greater_is_better=False
)
scores = cross_val_score(
    pipeline,
    data_train,
    target_train,
    cv=cv,
    scoring=scorer,
    n_jobs=-1,
)

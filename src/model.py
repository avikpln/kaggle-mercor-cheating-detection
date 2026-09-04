# -- Import Libraries --
import pandas as pd
import numpy as np

from data import load_train_data, load_social_graph
from imputation import get_imputer

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of cross-validation folds.
N_SPLITS = 5

# Flag to enable debug mode.
DEBUG = False

# Number of qualifying connected components to sample in debug mode.
DEBUG_N_COMPONENTS = 100

# Max iterations for label propagation in debug mode.
DEBUG_MAX_ITER = 1000

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
imputer = get_imputer(RANDOM_STATE)

# -- Preprocessing --
from sklearn.preprocessing import FunctionTransformer

preprocessor = FunctionTransformer()

# -- Label Propagation --
import networkx as nx
from sklearn.semi_supervised import LabelPropagation

class GraphLabelFiller:

    def __init__(self, graph, resolver=None):
        self.graph = graph
        self.resolver = resolver

    def fit_transform(self, X, y):
        # Calculate connected components.
        components = list(nx.connected_components(self.graph))

        # Get the user_hash values of labeled train users.
        y_labeled = set(y.dropna().index)

        # Keep components with at least one labeled train user.
        qualifying_components = [
            component for component in components
            if component & y_labeled
        ]

        if DEBUG:
            rng = np.random.RandomState(RANDOM_STATE)
            qualifying_components = rng.choice(
                qualifying_components,
                size=DEBUG_N_COMPONENTS,
                replace=False,
            ).tolist()

        # Retain only the nodes in the qualifying components.
        keep_nodes = set().union(*qualifying_components)
        graph = self.graph.subgraph(keep_nodes).copy()

        # Order nodes: qualifying train users first, then the rest.
        users_in_qualifying = set(X.index) & set(graph.nodes)

        # Sort for reproducibility.
        nodelist = (
            sorted(users_in_qualifying)
            + sorted(set(graph.nodes) - users_in_qualifying)
        )

        # Create the adjacency matrix for the qualifying graph.
        adjacency_matrix = nx.to_scipy_sparse_array(graph, nodelist=nodelist)

        # Build the label array aligned to nodelist, using -1 for unlabeled.
        graph_y = y.reindex(nodelist).fillna(-1).to_numpy()

        # Run label propagation using the precomputed adjacency
        # matrix as the kernel.
        kernel = lambda *args: adjacency_matrix
        max_iter = 10000
        if DEBUG:
            max_iter = DEBUG_MAX_ITER
        label_propagator = LabelPropagation(kernel=kernel, max_iter=max_iter)
        label_propagator.fit(adjacency_matrix, graph_y)

        # Recover propagated labels, aligned to nodelist.
        propagated_labels = pd.Series(
            label_propagator.transduction_, index=nodelist
        )

        # Fill unlabeled train users with their propagated labels.
        propagated_labels = propagated_labels.reindex(X.index)
        y = y.fillna(propagated_labels)

        # Resolve remaining unlabeled users.
        if self.resolver is None:
            X, y = X[y.notna()], y[y.notna()]
        else:
            X, y = self.resolver.process(X, y)

        return X, y

# Populate graph.
graph = nx.from_pandas_edgelist(
    social_graph,
    source="user_a",
    target="user_b",
)

# -- Resolution --
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

class EstimatorBasedResolver:

    def __init__(self, estimator):
        self.estimator = estimator

    def process(self, X, y):
        # Identify labeled and unlabeled samples.
        labeled = y.notna()
        unlabeled = y.isna()

        # Fit a clone of the estimator on the labeled samples.
        estimator = clone(self.estimator)
        estimator.fit(X[labeled], y[labeled])

        # Resolve the remaining labels using the fitted estimator.
        y = y.copy()
        y.loc[unlabeled] = estimator.predict(X[unlabeled])

        return X, y

resolution_estimator = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE
)
resolver = EstimatorBasedResolver(resolution_estimator)

# Create an instance of the GraphLabelFiller class.
label_filler = GraphLabelFiller(graph, resolver)

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

# -- Semi-Supervised Learning --
from sklearn.base import BaseEstimator, ClassifierMixin

class SemiSupervisedClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier, label_filler):
        self.classifier = classifier
        self.label_filler = label_filler

    def fit(self, X, y):
        X_filled, y_filled = self.label_filler.fit_transform(X, y)
        self.classifier.fit(X_filled, y_filled)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

estimator = SemiSupervisedClassifier(classifier, label_filler)

# -- Pipeline --
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("imputer", imputer),
    ("preprocessor", preprocessor),
    ("estimator", estimator),
])

# -- Training and Evaluation --
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

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

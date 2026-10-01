# -- Imports --
import time

from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from data import load_social_graph, load_train_data
from evaluation import LabeledStratifiedKFold
from graph_utils import directed_reachable
from imputation import get_miceforest_imputer

N_SPLITS = 5
RANDOM_STATE = 42

# -- Data --
def _load_data():
    data_train = load_train_data().set_index("user_hash")
    target_name = "is_cheating"
    y = data_train[target_name]
    X = data_train.drop(columns=[target_name])
    return X, y

# -- LabelPropagator --
class LabelPropagator:
    def __init__(self, graph=None, feature_kernel=None, resolver=None,
                 max_iter=10000, propagator_cls=LabelPropagation,
                 **propagator_kwargs):
        if (graph is None) == (feature_kernel is None):
            raise ValueError(
                "Exactly one of graph or feature_kernel must be set."
            )
        self.graph = graph
        self.feature_kernel = feature_kernel
        self.resolver = resolver
        self.max_iter = max_iter
        self.propagator_cls = propagator_cls
        self.propagator_kwargs = propagator_kwargs

    def fit_transform(self, X, y):
        if self.graph is not None:
            adjacency_matrix, graph_y, nodelist, unreached = (
                self._build_graph_kernel(X, y)
            )
            propagator = self.propagator_cls(
                kernel=lambda *args: adjacency_matrix,
                max_iter=self.max_iter,
                **self.propagator_kwargs,
            )
            propagator.fit(adjacency_matrix, graph_y)
            propagated_labels = pd.Series(
                propagator.transduction_, index=nodelist
            )
            if unreached:
                propagated_labels.loc[list(unreached)] = np.nan
        else:
            graph_y = y.fillna(-1).to_numpy()
            propagator = self.propagator_cls(
                kernel=self.feature_kernel,
                max_iter=self.max_iter,
                **self.propagator_kwargs,
            )
            propagator.fit(X, graph_y)
            propagated_labels = pd.Series(
                propagator.transduction_, index=X.index
            )
        propagated_labels = propagated_labels.reindex(X.index)
        y_filled = y.fillna(propagated_labels)
        if self.resolver is None:
            X, y_filled = X[y_filled.notna()], y_filled[y_filled.notna()]
        else:
            X, y_filled = self.resolver.process(X, y_filled)
        return X, y_filled

    def _build_graph_kernel(self, X, y):
        graph = self.graph
        if graph.is_directed():
            components = list(nx.weakly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))

        y_labeled = set(y.dropna().index)
        qualifying_components = [
            component for component in components
            if component & y_labeled
        ]
        keep_nodes = set().union(*qualifying_components)
        graph = graph.subgraph(keep_nodes).copy()

        users_in_qualifying = set(X.index) & set(graph.nodes)
        nodelist = (
            sorted(users_in_qualifying)
            + sorted(set(graph.nodes) - users_in_qualifying)
        )

        adjacency_matrix = nx.to_scipy_sparse_array(
            graph, nodelist=nodelist
        )

        unreached = set()
        if graph.is_directed():
            # sklearn propagates from columns to rows, while NetworkX
            # stores edges as source -> target. Transpose to propagate
            # labels from referrers to referred users.
            adjacency_matrix = adjacency_matrix.T
            reached = directed_reachable(graph, y_labeled)
            unreached = set(nodelist) - reached

        graph_y = y.reindex(nodelist).fillna(-1).to_numpy()
        return adjacency_matrix, graph_y, nodelist, unreached

# -- Testing --
def _run_fold(fold_idx, X, y, build_propagator, dev_positions,
              imputer_fn=None):
    print(f"Starting fold {fold_idx + 1}/{N_SPLITS}...")

    y_masked = y.copy()
    y_masked.iloc[dev_positions] = np.nan

    if imputer_fn is not None:
        train_positions = np.setdiff1d(
            np.arange(len(X)), dev_positions
        )
        imputer = imputer_fn()
        imputer.fit(X.iloc[train_positions])
        X_fold = imputer.transform(X)
        X_fold.index = X.index
    else:
        X_fold = X

    label_propagator = build_propagator()
    _, y_filled = label_propagator.fit_transform(X_fold, y_masked)

    dev_index = X.index[dev_positions]
    resolved_dev_index = dev_index.intersection(y_filled.index)
    y_true = y.loc[resolved_dev_index]
    y_pred = y_filled.loc[resolved_dev_index]

    resolved_fraction = len(resolved_dev_index) / len(dev_index)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print(f"Finished fold {fold_idx + 1}/{N_SPLITS}.")
    return resolved_fraction, balanced_accuracy, y_true, y_pred


def evaluate(X, y, build_propagator, note, imputer_fn=None):
    labeled_index = y.dropna().index

    cv = LabeledStratifiedKFold(
        N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    X_labeled = X.loc[labeled_index]
    y_labeled = y.loc[labeled_index]

    full_pos = {idx: pos for pos, idx in enumerate(X.index)}
    folds = []

    for _, dev_rel_positions in cv.split(X_labeled, y_labeled):
        dev_index = X_labeled.index[dev_rel_positions]
        dev_positions = np.array([full_pos[i] for i in dev_index])
        folds.append(dev_positions)

    start = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(_run_fold)(
            i, X, y, build_propagator, dev_positions, imputer_fn
        )
        for i, dev_positions in enumerate(folds)
    )
    elapsed = time.perf_counter() - start

    fold_resolved_fractions = [r[0] for r in results]
    fold_balanced_accuracies = [r[1] for r in results]
    overall_y_true = pd.concat([r[2] for r in results])
    overall_y_pred = pd.concat([r[3] for r in results])

    print(f"[{note}] ({elapsed:.1f}s)")
    print(f"Per-fold resolved fraction: {fold_resolved_fractions}")
    print(f"Per-fold balanced accuracy: {fold_balanced_accuracies}")
    print(f"Fold accuracy std: {np.std(fold_balanced_accuracies):.4f}")
    print(
        f"Overall balanced accuracy: "
        f"{balanced_accuracy_score(overall_y_true, overall_y_pred):.4f}"
    )

# -- Single-source baselines --
def test_baseline_propagation_undirected():
    X, y = _load_data()
    social_graph = load_social_graph()
    graph = nx.from_pandas_edgelist(
        social_graph, source="user_a", target="user_b",
    )
    evaluate(
        X, y,
        lambda: LabelPropagator(graph=graph),
        "LabelPropagation, topology only, undirected",
    )


def test_baseline_propagation_directed():
    X, y = _load_data()
    social_graph = load_social_graph()
    graph = nx.from_pandas_edgelist(
        social_graph, source="user_a", target="user_b",
        create_using=nx.DiGraph,
    )
    evaluate(
        X, y,
        lambda: LabelPropagator(graph=graph),
        "LabelPropagation, topology only, directed",
    )


def get_imputer():
    # imputer = IterativeImputer(random_state=RANDOM_STATE)
    imputer = SimpleImputer(strategy="mean")
    # imputer = get_miceforest_imputer(random_state=RANDOM_STATE)
    imputer.set_output(transform="pandas")
    return imputer


def test_baseline_propagation_knn():
    X, y = _load_data()
    evaluate(
        X, y,
        lambda: LabelPropagator(feature_kernel="knn"),
        "LabelPropagation, features only, KNN kernel",
        imputer_fn=get_imputer,
    )


def test_baseline_spreading_undirected():
    X, y = _load_data()
    social_graph = load_social_graph()
    graph = nx.from_pandas_edgelist(
        social_graph, source="user_a", target="user_b",
    )
    evaluate(
        X, y,
        lambda: LabelPropagator(
            graph=graph, propagator_cls=LabelSpreading
        ),
        "LabelSpreading, topology only, undirected",
    )


def test_baseline_spreading_directed():
    X, y = _load_data()
    social_graph = load_social_graph()
    graph = nx.from_pandas_edgelist(
        social_graph, source="user_a", target="user_b",
        create_using=nx.DiGraph,
    )
    evaluate(
        X, y,
        lambda: LabelPropagator(
            graph=graph, propagator_cls=LabelSpreading
        ),
        "LabelSpreading, topology only, directed",
    )


def test_baseline_spreading_knn():
    X, y = _load_data()
    evaluate(
        X, y,
        lambda: LabelPropagator(
            feature_kernel="knn", propagator_cls=LabelSpreading
        ),
        "LabelSpreading, features only, KNN kernel",
        imputer_fn=get_imputer,
    )


def test_baseline_propagation():
    test_baseline_propagation_undirected()
    test_baseline_propagation_directed()
    test_baseline_propagation_knn()


def test_baseline_spreading():
    test_baseline_spreading_undirected()
    test_baseline_spreading_directed()
    test_baseline_spreading_knn()


def test_baseline():
    test_baseline_propagation()
    test_baseline_spreading()

# -- Analysis Functions --
def quantify_directed_unreachable():
    data_train = load_train_data()
    social_graph = load_social_graph()
    graph = nx.from_pandas_edgelist(
        social_graph, source="user_a", target="user_b",
        create_using=nx.DiGraph,
    )
    graph_users = set(graph.nodes)
    labeled = set(
        data_train.loc[
            data_train["is_cheating"].notna(), "user_hash"
        ]
    )
    unlabeled = set(
        data_train.loc[
            data_train["is_cheating"].isna(), "user_hash"
        ]
    )
    reached = directed_reachable(graph, labeled)
    outside_graph = unlabeled - graph_users
    in_graph_unreached = (unlabeled & graph_users) - reached
    reachable = unlabeled & reached
    n = len(data_train)
    print(f"All train users: {n}")
    print(
        f"Unlabeled, outside graph: {len(outside_graph)} "
        f"({len(outside_graph) / n:.2%})"
    )
    print(
        f"Unlabeled, in graph, no directed path from labeled: "
        f"{len(in_graph_unreached)} "
        f"({len(in_graph_unreached) / n:.2%})"
    )
    print(
        f"Unlabeled, reachable via directed propagation: "
        f"{len(reachable)} ({len(reachable) / n:.2%})"
    )

# -- Main --
if __name__ == "__main__":
    test_baseline()
    quantify_directed_unreachable()

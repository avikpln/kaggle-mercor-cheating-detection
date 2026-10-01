import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.impute import SimpleImputer

from gnn_conv import ConvGNNClassifier
from gnn_one_hop import OneHopGNNClassifier

# -- Constants --

# Flag to enable debug mode.
DEBUG = False

# Number of qualifying connected components to sample in debug mode.
DEBUG_N_COMPONENTS = 100

# Max iterations for label propagation in debug mode.
DEBUG_MAX_ITER = 1000

# Decision threshold for the GNN's predicted probability of cheating.
GNN_THRESHOLD = 0.5


class EstimatorBasedResolver:

    def __init__(self, estimator):
        self.estimator = estimator

    def process(self, X, y):
        labeled = y.notna()
        unlabeled = y.isna()

        estimator = clone(self.estimator)
        estimator.fit(X[labeled], y[labeled])

        y = y.copy()
        y.loc[unlabeled] = estimator.predict(X[unlabeled])

        return X, y


class GraphLabelFiller:

    def __init__(self, resolver, random_state=None):
        self.resolver = resolver
        self.random_state = random_state

    def fit_transform(self, X, y):
        propagated_labels = self.fill_labels(X, y)
        y = y.fillna(propagated_labels)
        if self.resolver is None:
            X, y = X[y.notna()], y[y.notna()]
        else:
            X, y = self.resolver.process(X, y)
        return X, y

    def fill_labels(self, X, y):
        raise NotImplementedError


class GraphLabelFillerLP(GraphLabelFiller):

    def __init__(self, graph, resolver, random_state=None):
        super().__init__(resolver, random_state)
        self.graph = graph.to_undirected()

    def fill_labels(self, X, y):
        components = list(nx.connected_components(self.graph))
        y_labeled = set(y.dropna().index)

        qualifying_components = [
            component for component in components
            if component & y_labeled
        ]

        if DEBUG:
            rng = np.random.RandomState(self.random_state)
            qualifying_components = rng.choice(
                qualifying_components,
                size=DEBUG_N_COMPONENTS,
                replace=False,
            ).tolist()

        keep_nodes = set().union(*qualifying_components)
        graph = self.graph.subgraph(keep_nodes).copy()

        users_in_qualifying = set(X.index) & set(graph.nodes)
        nodelist = (
            sorted(users_in_qualifying)
            + sorted(set(graph.nodes) - users_in_qualifying)
        )

        adjacency_matrix = nx.to_scipy_sparse_array(graph, nodelist=nodelist)
        graph_y = y.reindex(nodelist).fillna(-1).to_numpy()

        kernel = lambda *args: adjacency_matrix
        max_iter = DEBUG_MAX_ITER if DEBUG else 10000
        label_propagator = LabelPropagation(kernel=kernel, max_iter=max_iter)
        label_propagator.fit(adjacency_matrix, graph_y)

        propagated_labels = pd.Series(
            label_propagator.transduction_, index=nodelist
        )
        return propagated_labels.reindex(X.index)


class GraphLabelFillerGNN(GraphLabelFiller):
    def __init__(self, directed_graph, resolver=None, random_state=None):
        super().__init__(resolver, random_state)
        self.graph_users = set(directed_graph.nodes)

    def fill_labels(self, X, y):
        graph_users = X.index.intersection(self.graph_users)
        X_graph = X.loc[graph_users]
        y_graph = y.loc[graph_users]

        labeled = y_graph.notna()
        X_labeled = X_graph.loc[labeled]
        y_labeled = y_graph.loc[labeled]
        X_unlabeled = X_graph.loc[~labeled]

        # model = OneHopGNNClassifier(random_state=self.random_state)
        model = ConvGNNClassifier(random_state=self.random_state)
        model.fit(X_labeled, y_labeled)
        probs = model.predict_proba(X_unlabeled)[:, 1]

        propagated_labels = pd.Series(
            (probs >= GNN_THRESHOLD).astype(int),
            index=X_unlabeled.index,
        )
        return propagated_labels


def get_labeler(graph, random_state=None):
    resolution_estimator = HistGradientBoostingClassifier(
        random_state=random_state
    )
    resolver = EstimatorBasedResolver(resolution_estimator)

    # return GraphLabelFillerLP(graph, resolver=None, random_state=random_state)
    return GraphLabelFillerGNN(graph, resolver=None, random_state=random_state)

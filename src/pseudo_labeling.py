import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.impute import SimpleImputer

from gnn import GNNClassifier

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

    def __init__(self, directed_graph, resolver, random_state=None):
        super().__init__(resolver, random_state)
        self.directed_graph = directed_graph
        self.undirected_graph = directed_graph.to_undirected()

    def _qualifying_nodes(self, y):
        labeled = set(y.dropna().index)
        keep = set()
        for component in nx.connected_components(self.undirected_graph):
            if component & labeled:
                keep |= component
        return keep

    def _compute_ghost_flags(self, graph, feature_index):
        feature_index = set(feature_index)
        rows = []
        for node in graph.nodes:
            if node in feature_index:
                rows.append((node, 0, 0))
                continue
            is_referrer = graph.out_degree(node) > 0
            is_referred = graph.in_degree(node) > 0
            rows.append((node, int(is_referrer), int(is_referred)))
        return pd.DataFrame(
            rows, columns=["user_hash", "is_ghost_referrer", "is_ghost_referred"]
        ).set_index("user_hash")

    def _normalize_adjacency(self, adjacency):
        adjacency = adjacency.tocsr().astype(np.float32)
        degree = np.asarray(adjacency.sum(axis=1)).flatten()
        degree[degree == 0] = 1.0
        return adjacency.multiply(1.0 / degree[:, None]).tocsr()

    def _scipy_to_torch_sparse(self, matrix):
        import torch
        matrix = matrix.tocoo()
        indices = torch.tensor(
            np.vstack([matrix.row, matrix.col]), dtype=torch.long
        )
        values = torch.tensor(matrix.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=matrix.shape).coalesce()

    def _build_relation_adjacency(self, directed_subgraph, nodelist):
        successor_adj = nx.to_scipy_sparse_array(
            directed_subgraph, nodelist=nodelist
        )
        predecessor_adj = successor_adj.T
        successor_adj = self._normalize_adjacency(successor_adj)
        predecessor_adj = self._normalize_adjacency(predecessor_adj)
        return (
            self._scipy_to_torch_sparse(predecessor_adj),
            self._scipy_to_torch_sparse(successor_adj),
        )

    def _impute_features(self, X_full, flags):
        is_ghost_referrer = flags["is_ghost_referrer"].to_numpy().astype(bool)
        is_ghost_referred = flags["is_ghost_referred"].to_numpy().astype(bool)
        referrer_only = is_ghost_referrer & ~is_ghost_referred
        real_users = ~is_ghost_referrer & ~is_ghost_referred

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X_full.loc[real_users])
        X_imputed = pd.DataFrame(
            imputer.transform(X_full), columns=X_full.columns, index=X_full.index
        )
        X_imputed.loc[referrer_only] = -1.0

        return pd.concat([X_imputed, flags], axis=1)

    def fill_labels(self, X, y):
        qualifying_nodes = self._qualifying_nodes(y)
        nodelist = sorted(qualifying_nodes)
        directed_subgraph = self.directed_graph.subgraph(qualifying_nodes)

        ghost_flags = self._compute_ghost_flags(directed_subgraph, X.index)
        X_full = X.reindex(nodelist)
        flags = ghost_flags.reindex(nodelist)
        y_full = y.reindex(nodelist)

        adj_pred, adj_succ = self._build_relation_adjacency(
            directed_subgraph, nodelist
        )
        X_features = self._impute_features(X_full, flags)

        model = GNNClassifier(adj_pred, adj_succ, self.random_state)
        model.fit(X_features.to_numpy(), y_full.to_numpy())
        probs = model.predict_proba(X_features.to_numpy())[:, 1]

        propagated_labels = pd.Series(
            (probs >= GNN_THRESHOLD).astype(int), index=nodelist
        )
        return propagated_labels.reindex(X.index)


def get_labeler(graph, random_state=None):
    resolution_estimator = HistGradientBoostingClassifier(
        random_state=random_state
    )
    resolver = EstimatorBasedResolver(resolution_estimator)

    # return GraphLabelFillerLP(graph, resolver=None, random_state=random_state)
    return GraphLabelFillerGNN(graph, resolver=None, random_state=random_state)

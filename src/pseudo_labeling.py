import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation

# Flag to enable debug mode.
DEBUG = False

# Number of qualifying connected components to sample in debug mode.
DEBUG_N_COMPONENTS = 100

# Max iterations for label propagation in debug mode.
DEBUG_MAX_ITER = 1000


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


class GraphLabelFiller:

    def __init__(self, graph, resolver=None, random_state=None):
        self.graph = graph
        self.resolver = resolver
        self.random_state = random_state

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
            rng = np.random.RandomState(self.random_state)
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


def get_labeler(graph, random_state=None):

    # Set up the resolution estimator.
    resolution_estimator = HistGradientBoostingClassifier(
        random_state=random_state
    )
    resolver = EstimatorBasedResolver(resolution_estimator)

    # Create an instance of the GraphLabelFiller class.
    label_filler = GraphLabelFiller(graph, resolver, random_state=random_state)

    return label_filler

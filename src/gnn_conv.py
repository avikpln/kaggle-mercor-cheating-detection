# -- Imports --
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from data import load_graph_components, load_social_graph

# -- Constants --

# Whether to include ghost users in the edge index.
INCLUDE_GHOSTS = False

# Small constant to avoid division by zero in scaling.
EPSILON = 1e-6

# -- Model --
class ConvGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes=2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, n_classes)

    def forward(self, x, edge_index):
        x = self.bn0(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# class ConvGNN(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes=2):
#         super().__init__()
#         self.bn0 = nn.BatchNorm1d(in_dim)
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.linear = nn.Linear(hidden_dim, n_classes)

#     def forward(self, x, edge_index):
#         x = self.bn0(x)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.bn1(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.bn2(x)
#         x = self.linear(x)
#         return x

# -- Classifier --
class ConvGNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None):
        self.hidden_dim = 128
        self.epochs = 200
        self.lr = 1e-2
        self.weight_decay = 0.0
        self.random_state = random_state

    def _seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

    def _screen(self, graph_components, X):
        train_users = set(X.index)
        qualifying_components = [
            comp for comp in graph_components
            if any(user in train_users for user in comp)
        ]
        return qualifying_components

    def _build_edge_index(self, X):
        graph_components = load_graph_components()
        qualifying_components = self._screen(graph_components, X)

        self.user_to_index = {user: i for i, user in enumerate(X.index)}
        self.size = len(X)
        def index_me(user):
            if user not in self.user_to_index:
                self.user_to_index[user] = self.size
                self.size += 1 
            return self.user_to_index[user]

        if INCLUDE_GHOSTS:
            self.edge_index = torch.tensor(
                np.array([
                    [index_me(u), index_me(v)]
                    for cc in qualifying_components
                    for u in cc for v in sorted(cc[u])  # for reproducibility
                ]),
                dtype=torch.long,
            ).T
        else:
            self.edge_index = torch.tensor(
                np.array([
                    [self.user_to_index[u], self.user_to_index[v]]
                    for cc in qualifying_components
                    for u in cc for v in sorted(cc[u])  # for reproducibility
                    if u in self.user_to_index and v in self.user_to_index
                ]),
                dtype=torch.long,
            ).T

    def _build_graph(self):
        self.graph = nx.from_pandas_edgelist(
            load_social_graph(),
            source="user_a",
            target="user_b",
            create_using=nx.DiGraph,
        )

    def _scale(self, X):
        return (X - self.min_) / (self.max_ - self.min_ + EPSILON)

    def _preprocess(self, X):
        self.min_, self.max_ = X.min(), X.max()
        mean = X.mean()
        X = self._scale(X)
        n_ghosts = self.size - len(X)
        ghosts = np.tile(self._scale(mean).to_numpy(), (n_ghosts, 1))
        return np.vstack([X.to_numpy(), ghosts])

    def _train(self, X, y):
        X = self._preprocess(X)
        self.X_train = X

        labeled = y.notna().to_numpy(copy=True)
        y_labeled = y[labeled]

        self.classes_ = np.unique(y_labeled).astype(int)
        self.model_ = ConvGNN(X.shape[1], self.hidden_dim, len(self.classes_))

        X = torch.tensor(X, dtype=torch.float32)
        y_labeled = torch.tensor(y_labeled.to_numpy(), dtype=torch.long)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model_.train()
            optimizer.zero_grad()
            logits = self.model_(X, self.edge_index)
            loss = loss_fn(logits[:len(y)][labeled], y_labeled)
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}: loss={loss.item():.4f}")

    def fit(self, X, y):
        self._seed()
        self._build_edge_index(X)
        self._build_graph()
        self._train(X, y)
        return self

    def _extend_users(self, X):
        ext_user_to_index = {}
        next_index = self.size
        graph_idx = []
        for user, row in X.iterrows():
            if user in self.user_to_index:
                graph_idx.append(self.user_to_index[user])
            elif user in ext_user_to_index:
                graph_idx.append(ext_user_to_index[user])
            else:
                ext_user_to_index[user] = next_index
                graph_idx.append(next_index)
                next_index += 1
        X_new = X[~X.index.isin(self.user_to_index)]
        X = np.vstack([self.X_train, X_new.to_numpy()])
        return X, ext_user_to_index, graph_idx

    def _extend_edge_index(self, ext_user_to_index):
        user_to_index = {**self.user_to_index, **ext_user_to_index}
        new_edges = [
            (u, v) for u, v in self.graph.edges()
            if (u in ext_user_to_index and v in user_to_index)
            or (v in ext_user_to_index and u in user_to_index)
        ]

        ext_edge_index = (
            torch.empty((2, 0), dtype=torch.long)
            if not new_edges
            else torch.tensor(
                [[user_to_index[u], user_to_index[v]] for u, v in new_edges],
                dtype=torch.long,
            ).T
        )

        return ext_edge_index
            
    def _extend(self, X):
        X, ext_user_to_index, graph_idx = self._extend_users(X)
        ext_edge_index = self._extend_edge_index(ext_user_to_index)
        return X, graph_idx, ext_edge_index

    def predict_proba(self, X):
        X = self._scale(X)
        X, graph_idx, ext_edge_index = self._extend(X)
        edge_index = torch.cat([self.edge_index, ext_edge_index], dim=1)

        X = torch.tensor(X, dtype=torch.float32)
        self.model_.eval()
        with torch.inference_mode():
            logits = self.model_(X, edge_index)
            probs = torch.softmax(logits[graph_idx], dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

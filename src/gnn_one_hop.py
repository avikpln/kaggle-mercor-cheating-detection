# -- Imports --
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_social_graph

# -- Constants --

# Small constant to avoid division by zero in scaling.
EPSILON = 1e-6

# -- Model --
class OneHopGNN(nn.Module):
    def __init__(self, in_dims, hidden_dim=64, n_classes=2):
        super().__init__()
        self_dim, neighbor_dim = in_dims
        self.self_lin = nn.Linear(self_dim, hidden_dim)
        self.pred_lin = nn.Linear(neighbor_dim, hidden_dim)
        self.succ_lin = nn.Linear(neighbor_dim, hidden_dim)
        head_in = hidden_dim * 3
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, X, X_pred, X_succ):
        self_h = self.self_lin(X)
        pred_h = self.pred_lin(X_pred)
        succ_h = self.succ_lin(X_succ)
        h = torch.cat([self_h, pred_h, succ_h], dim=-1)
        h = F.relu(h)
        return self.head(h)

# -- Classifier --
class OneHopGNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None):
        self.hidden_dim = 64
        self.epochs = 200
        self.lr = 1e-2
        self.weight_decay = 0.0
        self.random_state = random_state

    def _seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

    def _build_graph(self):
        self.graph = nx.from_pandas_edgelist(
            load_social_graph(),
            source="user_a",
            target="user_b",
            create_using=nx.DiGraph,
        )

    def _extract_neighbor_aggregates(self, X, context):
        users = set(X.index) | set(self.X_fit.index)

        for user in X.index:
            user_features = (
                self.X_fit.loc[user]
                if user in self.X_fit.index
                else X.loc[user]
            )

            for predecessor in self.graph.predecessors(user):
                edge = (predecessor, user)
                if edge in context["edge_seen"]:
                    continue
                context["edge_seen"].add(edge)

                if predecessor in users:
                    features = (
                        self.X_fit.loc[predecessor]
                        if predecessor in self.X_fit.index
                        else X.loc[predecessor]
                    )
                    context["X_pred"].loc[user] += features
                    context["real_in"].loc[user] += 1
                else:
                    context["ghost_in"].loc[user] += 1

                if predecessor in X.index:
                    context["X_succ"].loc[predecessor] += user_features
                    context["real_out"].loc[predecessor] += 1

            for successor in self.graph.successors(user):
                edge = (user, successor)
                if edge in context["edge_seen"]:
                    continue
                context["edge_seen"].add(edge)

                if successor in users:
                    features = (
                        self.X_fit.loc[successor]
                        if successor in self.X_fit.index
                        else X.loc[successor]
                    )
                    context["X_succ"].loc[user] += features
                    context["real_out"].loc[user] += 1
                else:
                    context["ghost_out"].loc[user] += 1

                if successor in X.index:
                    context["X_pred"].loc[successor] += user_features
                    context["real_in"].loc[successor] += 1

        return context

    def _initialize_neighbor_aggregates(self, X):
        self.X_fit = X
        self._context = {
            "edge_seen": set(),
            "real_in": pd.Series(0, index=X.index, dtype=int),
            "real_out": pd.Series(0, index=X.index, dtype=int),
            "X_pred": pd.DataFrame(0.0, index=X.index, columns=X.columns),
            "X_succ": pd.DataFrame(0.0, index=X.index, columns=X.columns),
            "ghost_in": pd.Series(0, index=X.index, dtype=int),
            "ghost_out": pd.Series(0, index=X.index, dtype=int),
        }

        self._context = self._extract_neighbor_aggregates(X, self._context)

        # Sanity check.
        assert (
            self._context["real_in"].sum() == self._context["real_out"].sum()
        )

    def _scale(self, X, min_, max_):
        return (X - min_) / (max_ - min_ + EPSILON)

    def _preprocess(self, X):
        ghost_in = self._context["ghost_in"]
        ghost_out = self._context["ghost_out"]
        X_pred = self._context["X_pred"].div(
            self._context["real_in"].clip(lower=1), axis=0
        )
        X_succ = self._context["X_succ"].div(
            self._context["real_out"].clip(lower=1), axis=0
        )

        X = pd.concat([X, ghost_in, ghost_out], axis=1)
        self.min_, self.max_ = X.min(), X.max()
        X = self._scale(X, self.min_, self.max_)

        self.pred_min_, self.pred_max_ = X_pred.min(), X_pred.max()
        X_pred = self._scale(X_pred, self.pred_min_, self.pred_max_)

        self.succ_min_, self.succ_max_ = X_succ.min(), X_succ.max()
        X_succ = self._scale(X_succ, self.succ_min_, self.succ_max_)

        return X, X_pred, X_succ

    def _train(self, X, y):
        X, X_pred, X_succ = self._preprocess(X)

        labeled = y.notna().to_numpy(copy=True)
        y_labeled = y[labeled]
        self.classes_ = np.unique(y_labeled).astype(int)

        in_dims = (X.shape[1], X_succ.shape[1])
        self.model_ = OneHopGNN(in_dims, self.hidden_dim, len(self.classes_))

        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        X_pred = torch.tensor(X_pred.to_numpy(), dtype=torch.float32)
        X_succ = torch.tensor(X_succ.to_numpy(), dtype=torch.float32)
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
            logits = self.model_(X, X_pred, X_succ)
            loss = loss_fn(logits[labeled], y_labeled)
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}: loss={loss.item():.4f}")

    def fit(self, X, y):
        self._seed()
        self._build_graph()
        self._initialize_neighbor_aggregates(X)
        self._train(X, y)
        return self

    def _extend_neighbor_aggregates(self, X):
        context = {
            "edge_seen": self._context["edge_seen"].copy(),
            "real_in": pd.Series(0, index=X.index, dtype=int),
            "real_out": pd.Series(0, index=X.index, dtype=int),
            "X_pred": pd.DataFrame(0.0, index=X.index, columns=X.columns),
            "X_succ": pd.DataFrame(0.0, index=X.index, columns=X.columns),
            "ghost_in": pd.Series(0, index=X.index, dtype=int),
            "ghost_out": pd.Series(0, index=X.index, dtype=int),
        }

        known_users = X.index.intersection(self.X_fit.index)
        for key in context.keys() - {"edge_seen"}:
            context[key].loc[known_users] = self._context[key].loc[known_users]

        context = self._extract_neighbor_aggregates(X, context)
        return (
            context["X_pred"].div(context["real_in"].clip(lower=1), axis=0),
            context["X_succ"].div(context["real_out"].clip(lower=1), axis=0),
            context["ghost_in"],
            context["ghost_out"],
        )

    def predict_proba(self, X):
        X_pred, X_succ, ghost_in, ghost_out = (
            self._extend_neighbor_aggregates(X)
        )
        X = pd.concat([X, ghost_in, ghost_out], axis=1)
        X = self._scale(X, self.min_, self.max_)
        X_pred = self._scale(X_pred, self.pred_min_, self.pred_max_)
        X_succ = self._scale(X_succ, self.succ_min_, self.succ_max_)

        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        X_pred = torch.tensor(X_pred.to_numpy(), dtype=torch.float32)
        X_succ = torch.tensor(X_succ.to_numpy(), dtype=torch.float32)
        self.model_.eval()
        with torch.inference_mode():
            logits = self.model_(X, X_pred, X_succ)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

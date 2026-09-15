# -- Imports --
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score

from data import load_social_graph, load_train_data
from evaluation import LabeledStratifiedKFold

N_SPLITS = 5
RANDOM_STATE = 42

# -- Data  --

def _load_data():
    data_train = load_train_data().set_index("user_hash")
    target_name = "is_cheating"
    y = data_train[target_name]
    X = data_train.drop(columns=[target_name])
    return X, y

# -- Ghost Node Utilities --

def _compute_ghost_flags(graph, feature_index):
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

# -- Graph Scope --

def _qualifying_nodes(undirected_graph, y):
    labeled = set(y.dropna().index)
    keep = set()
    for component in nx.connected_components(undirected_graph):
        if component & labeled:
            keep |= component
    return keep

# -- Adjacency Construction --

def _normalize_adjacency(adjacency, mode="mean"):
    adjacency = adjacency.tocsr().astype(np.float32)
    if mode == "sum":
        return adjacency
    if mode != "mean":
        raise NotImplementedError(f"Unsupported aggregation mode: {mode}")
    degree = np.asarray(adjacency.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    return adjacency.multiply(1.0 / degree[:, None]).tocsr()


def _scipy_to_torch_sparse(matrix):
    matrix = matrix.tocoo()
    indices = torch.tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long)
    values = torch.tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=matrix.shape).coalesce()


def build_relation_adjacency(directed_graph, nodelist, aggregation="mean"):
    successor_adj = nx.to_scipy_sparse_array(directed_graph, nodelist=nodelist)
    predecessor_adj = successor_adj.T
    successor_adj = _normalize_adjacency(successor_adj, aggregation)
    predecessor_adj = _normalize_adjacency(predecessor_adj, aggregation)
    return (
        _scipy_to_torch_sparse(predecessor_adj),
        _scipy_to_torch_sparse(successor_adj),
    )

# -- Feature Construction --

def build_node_features(nodelist, X, ghost_flags):
    return X.reindex(nodelist), ghost_flags.reindex(nodelist)


def impute_features(X_full, flags, train_positions, strategy="mean"):
    is_ghost_referred = flags["is_ghost_referred"].to_numpy().astype(bool)
    is_ghost_referrer = flags["is_ghost_referrer"].to_numpy().astype(bool)
    referrer_only = is_ghost_referrer & ~is_ghost_referred

    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_full.iloc[train_positions])
    X_imputed = pd.DataFrame(
        imputer.transform(X_full), columns=X_full.columns, index=X_full.index
    )
    X_imputed.iloc[referrer_only] = -1.0

    return pd.concat([X_imputed, flags], axis=1)

# -- Model --

class RelationalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, combine="concat", dropout=0.0):
        super().__init__()
        if combine not in ("concat", "sum"):
            raise ValueError(f"Unsupported combine mode: {combine}")
        self.combine = combine
        self.self_lin = nn.Linear(in_dim, hidden_dim)
        self.pred_lin = nn.Linear(in_dim, hidden_dim)
        self.succ_lin = nn.Linear(in_dim, hidden_dim)
        head_in = hidden_dim * 3 if combine == "concat" else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X, X_pred, X_succ):
        self_h = self.self_lin(X)
        pred_h = self.pred_lin(X_pred)
        succ_h = self.succ_lin(X_succ)
        if self.combine == "concat":
            h = torch.cat([self_h, pred_h, succ_h], dim=-1)
        else:
            h = self_h + pred_h + succ_h
        h = F.relu(h)
        return self.head(h).squeeze(-1)

# -- Training --

def train_gnn(model, X, X_pred, X_succ, y_train, train_mask,
               epochs=200, lr=1e-2, weight_decay=0.0):
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)

    n_pos = y_train_tensor[train_mask_tensor].sum().clamp(min=1.0)
    n_neg = (train_mask_tensor.sum() - n_pos).clamp(min=1.0)
    pos_weight = n_neg / n_pos

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X, X_pred, X_succ)
        loss = loss_fn(
            logits[train_mask_tensor], y_train_tensor[train_mask_tensor]
        )
        loss.backward()
        optimizer.step()

    return model

# -- Testing --

def _run_fold(fold_idx, X_full, flags, y_full, adj_pred, adj_succ,
              dev_positions, model_kwargs, epochs, lr, weight_decay):
    print(f"Starting fold {fold_idx + 1}/{N_SPLITS}...")

    y_masked = y_full.copy()
    y_masked.iloc[dev_positions] = np.nan

    all_positions = np.arange(len(X_full))
    train_positions = np.setdiff1d(all_positions, dev_positions)

    X_features = impute_features(X_full, flags, train_positions)
    X_tensor = torch.tensor(X_features.to_numpy(dtype=np.float32))
    X_pred = torch.sparse.mm(adj_pred, X_tensor)
    X_succ = torch.sparse.mm(adj_succ, X_tensor)

    train_mask = y_masked.notna().to_numpy()
    y_train_filled = y_masked.fillna(0).to_numpy(dtype=np.float32)

    model = RelationalGNN(in_dim=X_tensor.shape[1], **model_kwargs)
    train_gnn(
        model, X_tensor, X_pred, X_succ, y_train_filled, train_mask,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
    )

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor, X_pred, X_succ)
        preds = (torch.sigmoid(logits) >= 0.5).numpy().astype(int)

    y_true = y_full.iloc[dev_positions].to_numpy()
    y_pred = preds[dev_positions]

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"Finished fold {fold_idx + 1}/{N_SPLITS}.")
    return balanced_accuracy, y_true, y_pred


def evaluate(X, y, note, model_kwargs=None, epochs=200, lr=1e-3,
             weight_decay=0.0, aggregation="mean", n_jobs=-1):
    model_kwargs = model_kwargs or {}

    social_graph = load_social_graph()
    directed_graph = nx.from_pandas_edgelist(
        social_graph, source="user_a", target="user_b",
        create_using=nx.DiGraph,
    )
    undirected_graph = directed_graph.to_undirected()

    qualifying_nodes = _qualifying_nodes(undirected_graph, y)
    nodelist = sorted(qualifying_nodes)
    directed_subgraph = directed_graph.subgraph(qualifying_nodes)
    undirected_subgraph = undirected_graph.subgraph(qualifying_nodes)

    ghost_flags = _compute_ghost_flags(directed_subgraph, X.index)
    X_full, flags = build_node_features(nodelist, X, ghost_flags)
    y_full = y.reindex(nodelist)

    adj_pred, adj_succ = build_relation_adjacency(
        directed_subgraph, nodelist, aggregation
    )

    labeled_index = y_full.dropna().index
    node_positions = {node: pos for pos, node in enumerate(nodelist)}

    cv = LabeledStratifiedKFold(
        N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    X_labeled = X_full.loc[labeled_index]
    y_labeled = y_full.loc[labeled_index]

    folds = []
    for _, dev_rel_positions in cv.split(X_labeled, y_labeled):
        dev_index = X_labeled.index[dev_rel_positions]
        dev_positions = np.array([node_positions[i] for i in dev_index])
        folds.append(dev_positions)

    start = time.perf_counter()
    torch.manual_seed(RANDOM_STATE)
    results = [
        _run_fold(
            i, X_full, flags, y_full, adj_pred, adj_succ, dev_positions,
            model_kwargs, epochs, lr, weight_decay,
        )
        for i, dev_positions in enumerate(folds)
    ]
    elapsed = time.perf_counter() - start

    fold_balanced_accuracies = [r[0] for r in results]
    overall_y_true = np.concatenate([r[1] for r in results])
    overall_y_pred = np.concatenate([r[2] for r in results])

    print(f"[{note}] ({elapsed:.1f}s)")
    print(f"Per-fold balanced accuracy: {fold_balanced_accuracies}")
    print(f"Fold accuracy std: {np.std(fold_balanced_accuracies):.4f}")
    print(
        f"Overall balanced accuracy: "
        f"{balanced_accuracy_score(overall_y_true, overall_y_pred):.4f}"
    )

# -- Baselines --

def test_baseline_gnn():
    X, y = _load_data()
    evaluate(
        X, y,
        "GNN, mean aggregation",
        model_kwargs={"hidden_dim": 64, "combine": "concat", "dropout": 0.0},
        epochs=1000, lr=1e-2,
    )

# -- Main --

if __name__ == "__main__":
    test_baseline_gnn()

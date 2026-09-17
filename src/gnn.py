# -- Imports --
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# -- Model --

class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, combine, dropout):
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

# -- Classifier --

class GNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, adj_pred, adj_succ, random_state=None):
        self.adj_pred = adj_pred
        self.adj_succ = adj_succ
        self.hidden_dim = 64
        self.combine = "concat"
        self.dropout = 0.0
        self.epochs = 200
        self.lr = 1e-2
        self.weight_decay = 0.0
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
        if self.adj_pred.shape[0] != X_tensor.shape[0]:
            raise ValueError(
                f"adj_pred rows ({self.adj_pred.shape[0]}) must match "
                f"X rows ({X_tensor.shape[0]})"
            )
        if self.adj_succ.shape[0] != X_tensor.shape[0]:
            raise ValueError(
                f"adj_succ rows ({self.adj_succ.shape[0]}) must match "
                f"X rows ({X_tensor.shape[0]})"
            )

        X_pred = torch.sparse.mm(self.adj_pred, X_tensor)
        X_succ = torch.sparse.mm(self.adj_succ, X_tensor)

        y = np.asarray(y, dtype=np.float32)
        train_mask = ~np.isnan(y)
        y_filled = np.nan_to_num(y, nan=0.0)

        y_train_tensor = torch.tensor(y_filled, dtype=torch.float32)
        train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)

        n_pos = y_train_tensor[train_mask_tensor].sum().clamp(min=1.0)
        n_neg = (train_mask_tensor.sum() - n_pos).clamp(min=1.0)
        pos_weight = n_neg / n_pos

        model = GNNModel(X_tensor.shape[1], self.hidden_dim, self.combine,
                          self.dropout)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(X_tensor, X_pred, X_succ)
            loss = loss_fn(
                logits[train_mask_tensor], y_train_tensor[train_mask_tensor]
            )
            loss.backward()
            optimizer.step()

        self.model_ = model
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
        X_pred = torch.sparse.mm(self.adj_pred, X_tensor)
        X_succ = torch.sparse.mm(self.adj_succ, X_tensor)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor, X_pred, X_succ)
            probs_pos = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs_pos, probs_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# -- Factory --

def get_gnn(adj_pred, adj_succ):
    return GNNClassifier(adj_pred, adj_succ)

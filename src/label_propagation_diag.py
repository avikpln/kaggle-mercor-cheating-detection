from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from data import load_social_graph, load_train_data
from evaluation import LabeledStratifiedKFold
from pseudo_labeling import GraphLabelFiller

# Number of cross-validation folds.
N_SPLITS = 5

# Set random state for reproducibility.
RANDOM_STATE = 42

# -- Label Propagation Validation --

def _run_fold(fold_idx, X, y, graph, dev_positions, labeled_index):
    print(f"Starting fold {fold_idx + 1}/{N_SPLITS}...")

    dev_index = labeled_index[dev_positions]

    y_masked = y.copy()
    y_masked.loc[dev_index] = np.nan

    label_filler = GraphLabelFiller(graph, resolver=None)
    _, y_filled = label_filler.fit_transform(X, y_masked)

    resolved_dev_index = dev_index.intersection(y_filled.index)

    y_true = y.loc[resolved_dev_index]
    y_pred = y_filled.loc[resolved_dev_index]

    resolved_fraction = len(resolved_dev_index) / len(dev_index)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print(f"Finished fold {fold_idx + 1}/{N_SPLITS}.")
    return resolved_fraction, balanced_accuracy, y_true, y_pred


social_graph = load_social_graph()
graph = nx.from_pandas_edgelist(
    social_graph,
    source="user_a",
    target="user_b",
)

data_train = load_train_data().set_index("user_hash")
target_name = "is_cheating"
y = data_train[target_name]
X = data_train.drop(columns=[target_name])

labeled_index = y.dropna().index
labeled_X = X.loc[labeled_index]
labeled_y = y.loc[labeled_index]

cv = LabeledStratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

folds = list(cv.split(labeled_X, labeled_y))

results = Parallel(n_jobs=-1)(
    delayed(_run_fold)(i, X, y, graph, dev_positions, labeled_index)
    for i, (_, dev_positions) in enumerate(folds)
)

fold_resolved_fractions = [r[0] for r in results]
fold_balanced_accuracies = [r[1] for r in results]

overall_y_true = pd.concat([r[2] for r in results])
overall_y_pred = pd.concat([r[3] for r in results])

print(f"Per-fold resolved fraction: {fold_resolved_fractions}")
print(f"Per-fold balanced accuracy: {fold_balanced_accuracies}")
print(f"Fold accuracy std: {np.std(fold_balanced_accuracies):.4f}")

total_dev = sum(len(labeled_index[p]) for _, p in folds)

print(f"Overall resolved fraction: {len(overall_y_true) / total_dev:.4f}")

print(
    f"Overall balanced accuracy: "
    f"{balanced_accuracy_score(overall_y_true, overall_y_pred):.4f}"
)

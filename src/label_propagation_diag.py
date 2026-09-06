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

def _run_fold(fold_idx, X, y, graph, dev_positions, dev_candidate_index):
    print(f"Starting fold {fold_idx + 1}/{N_SPLITS}...")

    dev_index = dev_candidate_index[dev_positions]

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

components = list(nx.connected_components(graph))
y_labeled_index = set(y.dropna().index)

# Find components containing at least two labeled users.
eligible_components = [
    component
    for component in components
    if len(component & y_labeled_index) >= 2
]

# Reserve one labeled user in each eligible component as a permanent
# training anchor. All other labeled users in eligible components can
# be used as validation candidates.
rng = np.random.RandomState(RANDOM_STATE)
dev_candidate_users = []

for component in eligible_components:
    labeled_users = sorted(component & y_labeled_index)
    anchor = rng.choice(labeled_users)

    dev_candidate_users.extend(
        user for user in labeled_users if user != anchor
    )

dev_candidate_index = X.index[X.index.isin(dev_candidate_users)]

dev_candidates_X = X.loc[dev_candidate_index]
dev_candidates_y = y.loc[dev_candidate_index]

cv = LabeledStratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

folds = list(cv.split(dev_candidates_X, dev_candidates_y))

results = Parallel(n_jobs=-1)(
    delayed(_run_fold)(i, X, y, graph, dev_positions, dev_candidate_index)
    for i, (_, dev_positions) in enumerate(folds)
)

fold_resolved_fractions = [r[0] for r in results]
fold_balanced_accuracies = [r[1] for r in results]

overall_y_true = pd.concat([r[2] for r in results])
overall_y_pred = pd.concat([r[3] for r in results])

print(f"Per-fold resolved fraction: {fold_resolved_fractions}")
print(f"Per-fold balanced accuracy: {fold_balanced_accuracies}")
print(f"Fold accuracy std: {np.std(fold_balanced_accuracies):.4f}")

total_dev = sum(len(dev_candidate_index[p]) for _, p in folds)

print(f"Overall resolved fraction: {len(overall_y_true) / total_dev:.4f}")

print(
    f"Overall balanced accuracy: "
    f"{balanced_accuracy_score(overall_y_true, overall_y_pred):.4f}"
)

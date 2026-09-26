# -- Imports --
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from data import load_train_data
from evaluation import cost

# -- Constants --
RANDOM_STATE = 42
N_SPLITS = 5

# -- Data Preparation --
data_train = load_train_data().set_index("user_hash")

target_name = "is_cheating"
target_train = data_train[target_name]
data_train = data_train.drop(columns=[target_name])
data_train = data_train.drop(columns=["feature_014", "high_conf_clean"])

labeled = target_train.notna()
X, y = data_train[labeled], target_train[labeled]

# -- Training and Evaluation --
cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

scores = []
for train_idx, test_idx in cv.split(X, y):
    classifier = DummyClassifier(
        strategy="stratified", random_state=RANDOM_STATE
    )
    classifier.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred_proba = classifier.predict_proba(X.iloc[test_idx])
    scores.append(-cost(y.iloc[test_idx].to_numpy(), y_pred_proba))

print(np.array(scores))

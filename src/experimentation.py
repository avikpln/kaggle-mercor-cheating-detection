# -- Import Libraries --
import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

from data import load_train_data, load_social_graph
from evaluation import LabeledStratifiedKFold, get_scorer
from imputation import get_imputer
from pipeline import build_pipeline
from pseudo_labeling import get_labeler

# -- Constants --

# Set random state for reproducibility.
RANDOM_STATE = 42

# Number of cross-validation folds.
N_SPLITS = 5

# -- Data Preparation --

# Load the training data from a CSV file.
data_train = load_train_data().set_index("user_hash")

# Shuffle the training data.
data_train = data_train.sample(frac=1.0, random_state=RANDOM_STATE)

# Separate the target variable from the training data.
target_name = "is_cheating"
target_train = data_train[target_name]
data_train = data_train.drop(columns=[target_name])

# Feature selection.
data_train = data_train.drop(columns=["feature_014", "high_conf_clean"])

# Load social graph data from a CSV file.
social_graph = load_social_graph()

# -- Feature Imputation --
imputer = get_imputer(random_state=RANDOM_STATE)

# -- Preprocessing --
preprocessor = FunctionTransformer()

# -- Pseudo-Labeling --
graph = nx.from_pandas_edgelist(
    social_graph,
    source="user_a",
    target="user_b",
    create_using=nx.DiGraph,
)

labeler = get_labeler(graph, random_state=RANDOM_STATE)

# -- Classification --
classifier = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE,
    max_iter=100,
    learning_rate=0.1,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
)

# -- Pipeline --
pipeline = build_pipeline(imputer, preprocessor, labeler, classifier)

# -- Training and Evaluation --
cv = LabeledStratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(
    pipeline,
    data_train,
    target_train,
    cv=cv,
    scoring=get_scorer(),
    # n_jobs=-1,
)

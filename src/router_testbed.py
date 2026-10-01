# -- Imports --
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import SelfTrainingClassifier

from data import load_train_data, load_social_graph
from evaluation import LabeledStratifiedKFold, get_scorer
from imputation import get_imputer
from gnn_conv import ConvGNNClassifier
from router import RoutedClassifier

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
data_train = data_train.drop(columns=["high_conf_clean"])

# Load social graph data from a CSV file.
social_graph = load_social_graph()

# -- Classifier 0 --
class SelfTrainingClassifierNanSafe(SelfTrainingClassifier):
    def fit(self, X, y):
        y = y.fillna(-1).astype(int)
        return super().fit(X, y)

classifier0 = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE,
    max_iter=1000,
)
classifier0 = SelfTrainingClassifierNanSafe(classifier0)

# -- Classifier 1 --
classifier1 = ConvGNNClassifier(random_state=RANDOM_STATE)

# -- Predicate --
graph_users = set(social_graph["user_a"]) | set(social_graph["user_b"])
predicate = lambda user_hash: user_hash in graph_users

# -- Feature Imputation --
imputer1 = get_imputer(random_state=RANDOM_STATE)

# -- Pipelines --
pipelines = [
    make_pipeline(classifier0),
    make_pipeline(imputer1, classifier1),
]

# -- Classification --
model = RoutedClassifier(predicate, pipelines)

# -- Training and Evaluation --
cv = LabeledStratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print("Starting cross-validation...")

scores = cross_val_score(
    model,
    data_train,
    target_train,
    cv=cv,
    scoring=get_scorer(),
    verbose=2,
    # n_jobs=-1,
)

print("Cross-validation complete.")
print(scores)

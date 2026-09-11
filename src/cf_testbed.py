# -- Imports --
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC

from data import load_train_data, load_social_graph

# Set random state for reproducibility.
RANDOM_STATE = 42

# -- Data Loading and Preprocessing --

def _load_labeled_data(exclude_users=frozenset()):
    # Load the training data from a CSV file.
    data_train = load_train_data()

    # Shuffle the training data.
    data_train = data_train.sample(frac=1.0, random_state=RANDOM_STATE)

    # Separate the target variable from the training data.
    target_name = "is_cheating"
    target_train = data_train[target_name]
    data_train = data_train.drop(columns=[target_name])

    # Identify excluded users, to remove them below.
    excluded = data_train["user_hash"].isin(exclude_users)

    # Feature selection.
    data_train = data_train.drop(columns=["high_conf_clean", "user_hash"])

    # CF(x) = P(cheat | flagged): fit only on labeled (flagged)
    # examples, discarding the unlabeled population, as well as
    # any excluded users.
    labeled = target_train.notna() & ~excluded
    data_train, target_train = data_train[labeled], target_train[labeled]

    return data_train, target_train


def load_data():
    return _load_labeled_data()


def load_data_no_graph():
    social_graph = load_social_graph()
    graph_users = set(social_graph["user_a"]) | set(social_graph["user_b"])
    return _load_labeled_data(exclude_users=graph_users)


def load_data_no_referrers():
    social_graph = load_social_graph()
    referrers = set(social_graph["user_a"])
    return _load_labeled_data(exclude_users=referrers)


def load_data_no_referred():
    social_graph = load_social_graph()
    referred = set(social_graph["user_b"])
    return _load_labeled_data(exclude_users=referred)

# -- Model Definitions --

def get_model_HGBDT():
    model = HistGradientBoostingClassifier(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return model


def get_model_RF():
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
    )
    return model


def get_model_SVC():
    model = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        SVC(random_state=RANDOM_STATE),
    )
    return model


def get_model_LR():
    model = make_pipeline(
        IterativeImputer(),
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )
    return model


def get_model_NN():
    model = make_pipeline(
        IterativeImputer(),
        StandardScaler(),
        MLPClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )
    return model

# -- Tuning and Testing --
import time

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Number of cross-validation folds.
N_SPLITS = 5

cv = StratifiedKFold(
    N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def evaluate(model, X, y):
    return cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )


def run_test(name, get_model_fn, load_data_fn):
    X, y = load_data_fn()
    model = get_model_fn()

    start = time.perf_counter()
    scores = evaluate(model, X, y)
    elapsed = time.perf_counter() - start

    print(
        f"{name}: "
        f"{scores.mean():.4f} +/- {scores.std():.4f} "
        f"({elapsed:.1f}s)"
    )


ALL_MODELS = [
    ("HGBDT", get_model_HGBDT),
    ("RF", get_model_RF),
    ("SVC", get_model_SVC),
    ("LR", get_model_LR),
    ("NN", get_model_NN),
]

# -- Sweep: all labeled data --

def test_all_labeled():
    for name, get_model_fn in ALL_MODELS:
        run_test(name, get_model_fn, load_data)

# -- Sweep: labeled data, graph users excluded --

def test_no_graph():
    for name, get_model_fn in ALL_MODELS:
        run_test(name, get_model_fn, load_data_no_graph)

# -- HGBDT only: referrers excluded vs. referred excluded --

def test_referrers_vs_referred():
    run_test("HGBDT (no referrers)", get_model_HGBDT, load_data_no_referrers)
    run_test("HGBDT (no referred)", get_model_HGBDT, load_data_no_referred)

# -- Main --

if __name__ == "__main__":
    test_all_labeled()
    test_no_graph()
    test_referrers_vs_referred()

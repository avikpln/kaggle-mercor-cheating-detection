# -- Imports --
import os
import pickle
import sys

from sklearn.ensemble import HistGradientBoostingClassifier

from data import load_train_data

# Set random state for reproducibility.
RANDOM_STATE = 42

# Path to the cached flagger object.
FLAGGER_OBJECT_PATH = "src/flagger.pkl"

# -- Flagger Construction --

def load_data():
    # Load the training data from a CSV file.
    data_train = load_train_data()

    # Define the flagger target:
    # 1 if high_conf_clean is missing, 0 otherwise.
    data_train["flagged"] = data_train["high_conf_clean"].isna().astype(int)

    # Shuffle the training data.
    data_train = data_train.sample(frac=1.0, random_state=RANDOM_STATE)

    # Separate the target variable from the training data.
    target_name = "flagged"
    target_train = data_train[target_name]

    data_train = data_train.drop(columns=[target_name])

    # Feature selection.
    data_train = data_train.drop(
        columns=["high_conf_clean", "is_cheating", "user_hash"]
    )

    return data_train, target_train


def get_model():
    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_iter=300,
        learning_rate=1.0,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    return model


def build_flagger():
    X, y = load_data()
    model = get_model()
    model.fit(X, y)
    return model


def get_flagger():
    if os.path.exists(FLAGGER_OBJECT_PATH):
        with open(FLAGGER_OBJECT_PATH, "rb") as f:
            return pickle.load(f)

    flagger = build_flagger()

    with open(FLAGGER_OBJECT_PATH, "wb") as f:
        pickle.dump(flagger, f)

    return flagger

# -- Tuning and Testing --
import time

from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score, RandomizedSearchCV, StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer, PolynomialFeatures, StandardScaler,
)
from sklearn.svm import SVC

# Number of cross-validation folds.
N_SPLITS = 3

# Number of iterations for randomized search CV.
N_ITER = 20

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


def test_pipeline(pipeline, param_distributions, note=None):
    X, y = load_data()

    start = time.perf_counter()

    if param_distributions is None:
        scores = evaluate(pipeline, X, y)
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=N_ITER,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        scores = evaluate(search, X, y)

    elapsed = time.perf_counter() - start
    note = f" [{note}]" if note is not None else ""
    print(
        f"{pipeline.steps[-1][1].__class__.__name__}{note}: "
        f"{scores.mean():.4f} +/- {scores.std():.4f} "
        f"({elapsed:.1f}s)"
    )


def test_tree_model(tree_model, param_distributions):
    for imputer, note in zip(
        [
            IterativeImputer(),
            SimpleImputer(),
            FunctionTransformer(),
        ],
        [
            "iterative imputation, param search",
            "simple imputation, param search",
            "no imputation, param search",
        ],
    ):
        pipeline = make_pipeline(imputer, tree_model)
        test_pipeline(pipeline, param_distributions, note)


def test_hist_gradient_boosting():
    param_distributions = {
        "histgradientboostingclassifier__max_iter": randint(100, 501),
        "histgradientboostingclassifier__learning_rate": uniform(0.01, 0.49),
    }
    test_tree_model(HistGradientBoostingClassifier(), param_distributions)


def test_random_forest():
    param_distributions = {
        "randomforestclassifier__n_estimators": randint(100, 501),
        "randomforestclassifier__max_depth": [None, 10, 20, 30, 40],
        "randomforestclassifier__min_samples_leaf": randint(1, 21),
    }
    test_tree_model(RandomForestClassifier(), param_distributions)


def test_svc():
    for imputer, note in [
        (IterativeImputer(), "iterative imputation"),
        (SimpleImputer(), "simple imputation"),
    ]:
        pipeline = make_pipeline(
            imputer,
            StandardScaler(),
            SVC(),
        )
        test_pipeline(pipeline, None, note)

    for C, note in [
        (0.1, "iterative imputation, lower C"),
        (10.0, "iterative imputation, higher C"),
    ]:
        pipeline = make_pipeline(
            IterativeImputer(),
            StandardScaler(),
            SVC(C=C),
        )
        test_pipeline(pipeline, None, note)


def test_logistic_regression():
    for imputer, note in [
        (IterativeImputer(), "iterative imputation"),
        (SimpleImputer(), "simple imputation"),
    ]:
        pipeline = make_pipeline(
            imputer,
            StandardScaler(),
            LogisticRegression(),
        )
        test_pipeline(pipeline, None, note)

    for imputer, note in [
        (
            IterativeImputer(),
            "iterative imputation, poly features (deg=2)",
        ),
        (SimpleImputer(), "simple imputation, poly features (deg=2)"),
    ]:
        pipeline = make_pipeline(
            imputer,
            PolynomialFeatures(degree=2),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        test_pipeline(pipeline, None, note)

    param_distributions = {
        "logisticregression__C": loguniform(0.01, 100),
    }

    pipeline = make_pipeline(
        IterativeImputer(),
        PolynomialFeatures(degree=2),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    test_pipeline(
        pipeline,
        param_distributions,
        "iterative imputation, poly features, C search"
    )


def test_neural_network():
    for imputer, note in [
        (IterativeImputer(), "iterative imputation"),
        (SimpleImputer(), "simple imputation"),
    ]:
        pipeline = make_pipeline(
            imputer,
            StandardScaler(),
            MLPClassifier(max_iter=1000),
        )
        test_pipeline(pipeline, None, note)

    pipeline = make_pipeline(
        IterativeImputer(),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            max_iter=1000,
        ),
    )
    test_pipeline(pipeline, None, "iterative imputation, 2 hidden layers")


def test_all():
    test_hist_gradient_boosting()
    test_random_forest()
    test_svc()
    test_logistic_regression()
    test_neural_network()

# -- Validation Curve --
import matplotlib.pyplot as plt
from sklearn.model_selection import ValidationCurveDisplay

def plot_validation_curve():
    X, y = load_data()

    pipeline = make_pipeline(
        HistGradientBoostingClassifier(
            early_stopping=False,
        ),
    )

    ValidationCurveDisplay.from_estimator(
        pipeline,
        X,
        y,
        param_name="histgradientboostingclassifier__max_iter",
        param_range=range(100, 1001, 100),
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    plt.show()

# -- Main --

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"

    if command == "build":
        flagger = build_flagger()
        with open(FLAGGER_OBJECT_PATH, "wb") as f:
            pickle.dump(flagger, f)
    elif command == "test":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        if model == "HGBDT":
            test_hist_gradient_boosting()
        elif model == "RF":
            test_random_forest()
        elif model == "SVC":
            test_svc()
        elif model == "LR":
            test_logistic_regression()
        elif model == "NN":
            test_neural_network()
        elif model is None:
            test_all()
        else:
            raise ValueError(f"Unknown test: {model}")
    elif command == "curve":
        plot_validation_curve()
    else:
        raise ValueError(f"Unknown command: {command}")

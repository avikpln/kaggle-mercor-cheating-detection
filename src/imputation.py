# -- Imports -- 
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (
    IterativeImputer, MissingIndicator, SimpleImputer
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# -- Constants --
MISSING_INDICATOR_FEATURES = [
    "feature_001",
    "feature_002",
    "feature_004",
    "feature_005",
    "feature_006",
    "feature_007",
    "feature_008",
    "feature_011",
    "feature_012",
    "feature_014",
    "feature_017",
    "feature_018",
]

MISSING_INDICATOR_FEATURES_10PCT = [
    "feature_001",  # 11.77%
    "feature_002",  # 11.62%
    "feature_017",  # 10.55%
    "feature_018",  # 10.13%
]

# -- Missing Decorators --
def missing_indicators_parallel(indicator_columns):
    def decorator(imputer_factory):
        def wrapped(*args, **kwargs):
            imputer = imputer_factory(*args, **kwargs)
            transformer = ColumnTransformer(
                [
                    ("imputer", imputer, slice(None)),
                    ("missing", Pipeline([
                        ("indicator", MissingIndicator()),
                        ("numeric", FunctionTransformer(lambda X: X.astype(float))),
                    ]), indicator_columns),
                ],
                verbose_feature_names_out=False,
            )
            transformer.set_output(transform="pandas")
            return transformer

        return wrapped

    return decorator


def missing_indicators_series(indicator_columns):
    def decorator(imputer_factory):
        def wrapped(*args, **kwargs):
            imputer = imputer_factory(*args, **kwargs)
            indicators = ColumnTransformer(
                [
                    ("original", "passthrough", slice(None)),
                    ("missing", Pipeline([
                        ("indicator", MissingIndicator()),
                        ("numeric", FunctionTransformer(lambda X: X.astype(float))),
                    ]), indicator_columns),
                ],
                verbose_feature_names_out=False,
            )
            pipeline = Pipeline([
                ("indicators", indicators),
                ("imputer", imputer),
            ])
            pipeline.set_output(transform="pandas")
            return pipeline

        return wrapped

    return decorator

# -- Factory --
# @missing_indicators_parallel(MISSING_INDICATOR_FEATURES)
@missing_indicators_series(MISSING_INDICATOR_FEATURES)
# @missing_indicators_parallel(MISSING_INDICATOR_FEATURES_10PCT)
# @missing_indicators_series(MISSING_INDICATOR_FEATURES_10PCT)
def get_imputer(random_state=None):
    # imputer = SimpleImputer(strategy="mean")
    # imputer = SimpleImputer(strategy="median")
    # imputer = SimpleImputer(strategy="most_frequent")
    # imputer = SimpleImputer(strategy='constant', fill_value=0)
    # imputer = SimpleImputer(strategy='constant', fill_value=-1)

    imputer = IterativeImputer(random_state=random_state)

    # imputer = get_miceforest_imputer(random_state=random_state)

    imputer.set_output(transform="pandas")
    return imputer

# -- MICEForest Testing --
import pandas as pd
from miceforest import ImputationKernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

class MiceForestImputer(BaseEstimator, TransformerMixin):
    def __init__(self, iterations=5, random_state=None):
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.index_ = X.index  # save original index
        X = X.reset_index(drop=True)

        self.columns_ = X.columns
        self.kernel_ = ImputationKernel(
            X,
            num_datasets=1,
            random_state=self.random_state,
        )
        self.kernel_.mice(self.iterations)
        return self

    def transform(self, X):
        original_index = X.index  # save caller's index 
        X = pd.DataFrame(X, columns=self.columns_).reset_index(drop=True)

        imputed = self.kernel_.impute_new_data(
            X,
            datasets=[0],
            random_state=self.random_state,
        )
        result = imputed.complete_data(dataset=0)

        result.index = original_index  # reattach
        return result

    def get_feature_names_out(self, input_features=None):
        return self.columns_.to_numpy()


def get_miceforest_imputer(iterations=5, random_state=None):
    imputer = MiceForestImputer(
        iterations=iterations,
        random_state=random_state,
    )
    imputer.set_output(transform="pandas")
    return imputer


def test_miceforest():
    from flagger import RANDOM_STATE, test_pipeline

    for imputer, note in [
        (get_imputer(RANDOM_STATE), "iterative imputation"),
        (
            get_miceforest_imputer(
                iterations=20,
                random_state=RANDOM_STATE,
            ),
            "miceforest imputation",
        ),
    ]:
        pipeline = make_pipeline(
            imputer,
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        )
        test_pipeline(pipeline, None, note)

# -- Main --
if __name__ == "__main__":
    test_miceforest()

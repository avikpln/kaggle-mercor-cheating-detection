from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def get_imputer(random_state):
    imputer = IterativeImputer(random_state=random_state)
    imputer.set_output(transform="pandas")
    return imputer

# -- Imports --
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# -- Routed Classification --
class RoutedClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, predicate, pipelines):
        self.predicate = predicate
        self.pipelines = pipelines

    def _route(self, X):
        return X.index.to_series().map(self.predicate).to_numpy()

    def fit(self, X, y):
        mask = self._route(X)

        self.pipelines_ = [clone(pipeline) for pipeline in self.pipelines]

        self.pipelines_[0].fit(X[~mask], y[~mask])
        self.pipelines_[1].fit(X[mask], y[mask])

        self.classes_ = np.unique(y[~pd.isna(y)])

        return self

    def predict_proba(self, X):
        mask = self._route(X)

        probs = np.empty((len(X), len(self.classes_)), dtype=float)
        if (~mask).any():
            probs[~mask] = self.pipelines_[0].predict_proba(X[~mask])
        if mask.any():
            probs[mask] = self.pipelines_[1].predict_proba(X[mask])

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
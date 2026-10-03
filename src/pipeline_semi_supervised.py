# -- Imports --
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

# -- Semi-Supervised Classifier --
class SemiSupervisedClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier, labeler):
        self.classifier = classifier
        self.labeler = labeler

    def fit(self, X, y):
        X_filled, y_filled = self.labeler.fit_transform(X, y)
        self.classifier.fit(X_filled, y_filled)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

# -- Pipeline --
def get_ss_pipeline(imputer, preprocessor, labeler, classifier):
    estimator = SemiSupervisedClassifier(classifier, labeler)

    pipeline = Pipeline([
        ("imputer", imputer),
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])

    return pipeline

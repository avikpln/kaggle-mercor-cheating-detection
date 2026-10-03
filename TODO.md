## Data Analysis

- Apply the flagger to the test set (and to the unlabeled train users) to
  measure how flag-like they are, as a way to characterize the CV/test gap.
  See insight (4) in Section **Testing the flagger** of `discussion.ipynb`.

- Cheat/clean ratio: inspect the ratio of cheating to clean users among
  labeled referred users, and likewise among labeled referrers.

## Error Analysis

- Inspect underfitting and overfitting in the baseline pipeline and its main
  components, and apply appropriate techniques to reduce error (e.g.,
  regularization).

- Confusion matrix: inspect the baseline's confusion matrix to understand
  what the data-side causes of failure are.

## Experimentation

- Undirected GNN: test the baseline GNN with an undirected graph.

- One-hot encoding: exploratory experiment on the ordinal categorical
  features. No strong expectation, since trees are insensitive to it and the
  features are not nominal.

- Propensity weighting: treat flagging as selection bias, model the
  probability of being flagged from the features, and use it to reweight the
  cheating classifier toward the general population. Requires deciding how to
  evaluate it, since the validation data is flagged-only.

- PU learning: model the unlabeled users as a mixture of positive and
  negative examples (instead of pseudo-labeling them), as listed in the
  missing labels section of `design.ipynb`.

## Feature Selection

- Leave-one-feature-out ablation: identify informative features (those that
  carry signal) and uninformative or harmful ones. First step toward a broader
  feature selection mechanism.

## Feature Engineering

- Polynomial features: test whether interaction and power terms improve
  performance.

- Feature transformations: test transformations of the features (e.g., power
  transformations for the right-skewed ones).

## Hyperparameter Tuning

- Design and carry out a grid and/or randomized search for the hyperparameters
  of the baseline pipeline (or the current best performing one).

## Infrastructure

- Timing: measure and report running time for the baseline pipeline and other
  key components.

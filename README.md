# Cheating Detection — Mercor Kaggle Challenge

## Overview

This repo documents my work on the
**Mercor Cheating Detection Competition** (Kaggle), a challenge to
predict whether a candidate is engaging in cheating behavior during an
online interview, using anonymized behavioral features, platform
activity signals, and a social graph of relationships between users.

The dataset combines a small set of manually-reviewed labeled candidates
with a large pool of unlabeled (and partially high-confidence-clean)
examples — a semi-supervised, sampling-biased setup meant to mirror real
fraud-detection conditions.

See the
[Mercor Cheating Detection Competition on Kaggle]
(https://www.kaggle.com/competitions/mercor-cheating-detection) for the
full brief.

## Vision

The primary goal of this iteration is to substantially improve on the
v0 baseline by replacing an under-scoped first attempt with a properly
structured process: real exploratory data analysis (EDA), a model
family suited to small, tabular data with a high ratio of missing
values, and using the social graph and the semi-supervised structure
while tuning decision thresholds directly against the official Kaggle
evaluation cost metric.

## Roadmap

- [x] **EDA.** Data structure, feature relationships, missingness, sampling
bias, social graph analysis
- [x] **Design.** Modeling approach, imputation strategy, graph usage,
cost-aware threshold optimization
- [ ] **Implementation.** Build and validate models against the actual cost
metric
  - [x] End-to-end baseline pipeline
  - [ ] Enhance results
- [ ] **Report.** Test the final model on the holdout set.

## Data

Download the competition data with:

```bash
kaggle competitions download -c mercor-cheating-detection
```

For details on the data fields, format, and structure, see the
[competition data page on Kaggle](https://www.kaggle.com/competitions/mercor-cheating-detection/data).

## Submission (v0 Baseline)

- **Score:** `-1,863,965.00`
- **1st place:** `-1,463,180.00`
- **Score after rework:** TBD

## Repo Structure

```
kaggle-mercor-cheating-detection/
├── docs/
│   ├── Data.pdf                       # Official competition data reference
│   └── Overview.pdf                   # Official competition brief
├── notebooks/
│   ├── cheating-detection-eval.ipynb  # Official evaluation metric
│   ├── design.ipynb                   # Modeling design decisions
│   ├── discussion.ipynb               # Modeling discussion and experiments
│   └── eda.ipynb                      # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── baseline.py                    # Baseline model and results
│   ├── cf_testbed.py                  # Cheating | Flagged testbed
│   ├── classifier_testbed.py          # Classifier testbed
│   ├── data.py                        # Data loading
│   ├── dummy.py                       # Dummy baseline
│   ├── evaluation.py                  # Evaluation
│   ├── flagger.py                     # Flagging model
│   ├── graph_preprocessing.py         # Graph preprocessing and caching
│   ├── graph_utils.py                 # Graph utilities
│   ├── gnn_conv.py                    # Convolutional GNN
│   ├── gnn_one_hop.py                 # One-hop GNN model
│   ├── gnn_testbed.py                 # GNN testbed
│   ├── holdout.py                     # Holdout split
│   ├── imputation.py                  # Feature imputation
│   ├── label_propagation_diag.py      # Label propagation diagnostics
│   ├── label_propagation_testbed.py   # Label propagation testbed
│   ├── pipeline_semi_supervised.py    # Semi-supervised pipeline
│   ├── pseudo_labeling.py             # Pseudo-labeling
│   ├── router.py                      # Routed classifier
│   ├── router_testbed.py              # Routed classification testbed
│   ├── sanity.py                      # Sanity checks
│   └── TODO.md                        # Tasks to perform
├── .gitignore
├── LICENSE
└── README.md
```

## License

See [`LICENSE`](LICENSE).

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

## Data

Download the competition data with:

```bash
kaggle competitions download -c mercor-cheating-detection
```

For details on the data fields, format, and structure, see the
[competition data page on Kaggle](https://www.kaggle.com/competitions/mercor-cheating-detection/data).

## Submission (v0 Baseline)

- **Score:** `-1,863,965.00000`
- **1st place:** `-1,463,180.00`
- **Score after rework:** TBD

## Repo Structure

```
kaggle-mercor-cheating-detection/
├── docs/
│   ├── Data.pdf            # Official competition data reference
│   └── Overview.pdf        # Official competition brief
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── .gitignore
├── LICENSE
└── README.md
```

## License

See [`LICENSE`](LICENSE).

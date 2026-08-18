# Cheating Detection — Mercor Kaggle Challenge

## Overview

This repo contains my submission to the **Mercor Cheating Detection Competition** (Kaggle), a challenge to predict whether a candidate is engaging in cheating behavior during an online interview, using anonymized behavioral features, platform activity signals, and a social graph of relationships between users.

The dataset combined a small set of manually-reviewed labeled candidates with a large pool of unlabeled (and partially high-confidence-clean) examples — a semi-supervised, sampling-biased setup meant to mirror real fraud-detection conditions.

See the [Mercor Cheating Detection Competition on Kaggle](<https://www.kaggle.com/competitions/mercor-cheating-detection>) for the full brief.

## Status: 🚧 Baseline committed, actively being reworked

This was my **first serious ML competition submission**, done ~6 months ago. The version in this repo (tag `v0-baseline`) is my original, unmodified submission — warts and all. I'm now rebuilding it from the ground up as a learning project, and documenting the process publicly.

**Original result:**
- My score: `-1,863,965.00000`
- Leaderboard 1st place: `-1,463,180.00`

## Evaluation Metric (why this competition is harder than plain classification)

Submissions aren't scored on accuracy or AUC — they're scored on a **cost-based operational metric**. Predicted probabilities get mapped into three decision bands (auto-pass / manual review / auto-block), and the leaderboard score is the negative of the minimum achievable total cost, optimized over thresholds:

| Outcome | Cost |
|---|---|
| False Negative (cheater passes through) | $600 |
| False Positive, auto-block | $300 |
| False Positive, manual review | $150 |
| True Positive, manual review | $5 |
| Correct auto-pass / auto-block | $0 |

The asymmetry matters a lot: missing a cheater is 2–4x worse than any other kind of mistake, and a correct decision made via "just send everyone to manual review" is only cheap, not free.

## What went wrong the first time (honest retro)

Looking back at the baseline with fresh eyes, the main issues were:

1. **No real EDA.** I moved to modeling before understanding class balance, missingness patterns, or what the social graph actually looked like.
2. **Model choice driven by enthusiasm, not evidence.** I reached for deep learning on a small, tabular, heavily-missing-data problem — a setting where gradient-boosted trees (XGBoost/LightGBM/CatBoost) are usually the stronger default.
3. **Metric mismatch.** I optimized against coarse metrics like accuracy instead of the competition's actual cost-based objective. This alone likely explains a large chunk of the gap to the leaderboard.
4. **Semi-supervised structure mostly ignored.** The unlabeled pool and `high_conf_clean` flag are there for a reason (sampling bias correction, pseudo-labeling, etc.) — I don't believe I made deliberate use of them.
5. *(more to be added as I re-audit the code)*

## Roadmap

- [ ] **EDA** — class balance, missingness, feature distributions, graph structure/connectivity, label-vs-unlabeled population differences
- [ ] **Strategy** — define modeling approach explicitly (labeling strategy for unlabeled data, model family, graph feature usage, cost-aware threshold optimization) *before* writing model code
- [ ] **Modeling** — build, validate, and iterate against the actual cost metric, not a proxy
- [ ] **Writeup** — final results, what changed, what the cost-metric-driven approach bought over the naive baseline

Progress notes and decisions are tracked in [`PROJECT_LOG.md`](PROJECT_LOG.md).

## Repo structure

```
kaggle-mercor-cheating-detection/
├── LICENSE
├── Overview.pdf
├── PROJECT_LOG.md
└── README.md
```

## License

`<choose one, e.g. MIT>`

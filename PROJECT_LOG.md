# Project Log — Cheating Detection Rework

> Working memory for this project. Paste this file into a new chat to resume with full context.
> Update after each significant session. Keep entries dated and terse — this is a log, not prose.

---

## Competition facts (don't re-derive these)

- **Task:** predict `is_cheating` probability per candidate.
- **Data:** `feature_001`–`feature_018` (anonymized behavioral/activity features), `high_conf_clean` flag (unlabeled, high-confidence NOT cheating), `is_cheating` label (manually reviewed subset only), plus a **social graph** of relationships between users.
- **Train set:** mix of labeled + large unlabeled pool. **Test/holdout:** labeled only.
- **Core difficulty:** semi-supervised + sampling bias (labeled cases come from already-flagged candidates, not a random sample) + graph-based signal.
- **Metric:** NOT accuracy/AUC. Cost-based, three decision regions (auto-pass / manual review / auto-block), thresholds chosen to minimize total cost. Leaderboard score = negative of minimum total cost.

**Cost table:**
| Outcome | Cost |
|---|---|
| False Negative | $600 |
| False Positive — auto-block | $300 |
| False Positive — manual review | $150 |
| True Positive — manual review | $5 |
| Correct auto-pass/auto-block | $0 |

Implication: recall on true cheaters is paramount; false positives are tolerable in manual review, expensive in auto-block. A good model should be evaluated by simulating the threshold search, not by AUC alone.

## Baseline (v0) result

- My score: `<fill in>`
- 1st place: -1,463,180.00
- Rank: 403/419
- Self-diagnosed issues (from memory, 6 months later — may be incomplete):
  1. No systematic EDA
  2. Jumped to deep learning instead of trying GBTs (XGBoost etc.) first
  3. Optimized coarse metrics (e.g. accuracy) instead of the actual cost metric
  4. Unclear whether unlabeled data / `high_conf_clean` was used deliberately (semi-supervised techniques, sampling-bias correction) — needs re-checking against actual v0 code

## Roles / working agreement

- User writes all code and does the actual EDA/modeling. Claude acts as guide/reviewer/sounding board, not implementer, unless explicitly asked to write code.
- User will often use separate chats for EDA (to save resources) — this log is the handoff mechanism.
- Claude should ask "what do you think?" before opinions, and push back rather than rubber-stamp.

## Status

- [x] Read challenge instructions
- [x] Drafted README.md + this log
- [ ] Baseline code committed to GitHub as `v0-baseline` (user's action — Claude has no push access)
- [ ] EDA started
- [ ] Strategy doc written
- [ ] Modeling started

## Decision log

*(newest first — add an entry each session: date, decision, why, what's still open)*

- **2026-07-03** — Set up repo skeleton (README + this log) before touching code. Decided to keep public README high-level and put all working detail/history here. Cost table and known v0 weaknesses captured for reference. Next: user commits existing v0 code as-is, then starts EDA in a separate chat.

## Open questions

- What was my actual v0 leaderboard score? (fill in)
- Does the v0 code touch the social graph at all, or was it feature-only?
- What does the unlabeled pool size look like relative to labeled data? (will know after EDA)
- License choice for the repo?

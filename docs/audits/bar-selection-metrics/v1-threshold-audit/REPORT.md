---
title: "ODB Bar-Selection Metrics: Threshold-Based Statistical Audit (v1)"
date: 2026-03-14
version: 1
type: statistical-audit
symbol: BTCUSDT
threshold_dbps: 250
threshold_label: BPR25
bars_loaded: 10000
windows_sampled: 3200
seed: 42
data_range: "2025-12-18 → 2026-03-14"
key_finding: "conviction↔absorption ρ=−1.00 — they are mathematical inverses; conv_absorp_contest fires 0% and must be removed"
best_forward_predictor: conviction
best_forward_auc: 0.521
script: audit.py
---

# Flowsurface ODB Bar-Selection Metrics — Statistical Audit v1

**Data**: BPR25 BTCUSDT | **Bars**: 10,000 | **Windows sampled**: 3,200 | **Date range**: 2025-12-18 → 2026-03-14

> **Context**: This is the first audit of the `draw_bar_selection_stats` function in `src/chart/kline/bar_selection.rs`. It tests binary threshold signals (DIVERGES, SPLIT, CONTESTED) and uses standard Spearman ρ + triple-barrier methodology. The key actionable finding — `conv_absorp_contest` is logically dead — led directly to the v2 rank-based redesign.

---

## Executive Summary

**Key findings:**

- **Data quality**: 10,000 BPR25 bars loaded; 3,200 valid windows computed across log-uniform size distribution (10–500 bars).
- **Redundant metrics**: conviction↔absorption (ρ=−1.00) — these pairs carry near-identical information (|ρ|>0.85).
- **Predictive validity**: Best forward-return predictor is `conviction` (|ρ|=0.035). All metrics show weak-to-moderate predictive signal (|ρ|<0.15 typical for microstructure).
- **Regime robustness**: No metrics are severely regime-sensitive (all KS≤0.3), suggesting thresholds are reasonably portable across volatility regimes.
- **Edge case audit**: Equal-intensity AUC=0.4962 (expected 0.5). All-up windows: 0, all-dn: 0. NaN propagation is correct and bounded.

---

## A) Metric Distributions

| Metric         | Mean    | Std    | p5      | p25     | p50    | p75    | p95    | Min     | Max    | NaN# |
| -------------- | ------- | ------ | ------- | ------- | ------ | ------ | ------ | ------- | ------ | ---- |
| iwds           | −0.0035 | 0.3494 | −0.7050 | −0.1627 | 0.0227 | 0.1861 | 0.5707 | −0.9902 | 0.9837 | 0    |
| auc            | 0.5262  | 0.0896 | 0.3922  | 0.4840  | 0.5214 | 0.5652 | 0.6875 | 0.1111  | 1.0000 | 0    |
| log2_ratio     | −0.0276 | 1.2408 | −2.3247 | −0.2105 | 0.0703 | 0.3268 | 1.3610 | −6.7112 | 6.7065 | 0    |
| conviction     | 1.0270  | 0.2379 | 0.7200  | 0.9174  | 1.0021 | 1.0932 | 1.3819 | 0.4054  | 3.7500 | 0    |
| absorption     | 0.4998  | 0.0594 | 0.4061  | 0.4755  | 0.4994 | 0.5238 | 0.6000 | 0.1667  | 0.8889 | 0    |
| climax_up_frac | 0.5292  | 0.1569 | 0.2724  | 0.4444  | 0.5263 | 0.6154 | 0.7692 | 0.0000  | 1.0000 | 0    |

> **Note on absorption**: Removed in v2. At ρ=−1.00 with `conviction` on the rank-normalised scale, it is purely redundant. High conviction → low absorption by mathematical construction, not market information.

---

## B) Divergence Signal Base Rates by Window Size

| Window  | N   | climax_div% | urgency_split% | conv_absorp% |
| ------- | --- | ----------- | -------------- | ------------ |
| 10–30   | 925 | 28.2%       | 41.1%          | 0.0%         |
| 30–75   | 742 | 27.1%       | 37.9%          | 0.0%         |
| 75–150  | 573 | 31.9%       | 37.3%          | 0.0%         |
| 150–300 | 571 | 34.9%       | 39.4%          | 0.0%         |
| 300–500 | 389 | 36.2%       | 38.6%          | 0.0%         |

**Interpretation**: Signals are healthy if base rate is 5%–35% per window bucket.

- `< 2%`: signal too rare to be actionable
- `> 40%`: signal fires too often, likely noisy
- `conv_absorp_contest = 0.0%` across all buckets confirms this signal is **logically dead** and was removed in v2.

---

## C) Spearman Correlation Matrix

|                | iwds  | auc   | log2_ratio | conviction | absorption | climax_up_frac |
| -------------- | ----- | ----- | ---------- | ---------- | ---------- | -------------- |
| iwds           | +1.00 | +0.24 | +0.83      | +0.14      | −0.14      | +0.58          |
| auc            | +0.24 | +1.00 | +0.45      | −0.01      | +0.02      | +0.58          |
| log2_ratio     | +0.83 | +0.45 | +1.00      | +0.07      | −0.06      | +0.49          |
| conviction     | +0.14 | −0.01 | +0.07      | +1.00      | −1.00      | +0.10          |
| absorption     | −0.14 | +0.02 | −0.06      | −1.00      | +1.00      | −0.09          |
| climax_up_frac | +0.58 | +0.58 | +0.49      | +0.10      | −0.09      | +1.00          |

**Redundant pairs (|ρ|>0.85):**

- `conviction` ↔ `absorption`: ρ=−0.999 — mathematical inverses on rank-normalised scale.
- `iwds` ↔ `log2_ratio`: ρ=+0.83 — both measure raw intensity advantage of up vs down bars. Confirmed as collinear in v2 (ρ=+0.827 on 5,000 windows).

---

## D) Predictive Validity (Spearman ρ with Forward Returns)

| Metric         | fwd_5r  | fwd_10r | fwd_20r | fwd_up5 | best \|ρ\| |
| -------------- | ------- | ------- | ------- | ------- | ---------- |
| iwds           | +0.0053 | +0.0099 | +0.0117 | +0.0232 | 0.0232     |
| auc            | +0.0017 | +0.0092 | −0.0168 | +0.0027 | 0.0168     |
| log2_ratio     | +0.0104 | +0.0131 | +0.0048 | +0.0224 | 0.0224     |
| conviction     | −0.0087 | +0.0218 | +0.0355 | −0.0067 | 0.0355     |
| absorption     | +0.0089 | −0.0214 | −0.0352 | +0.0068 | 0.0352     |
| climax_up_frac | +0.0035 | +0.0017 | −0.0067 | +0.0121 | 0.0121     |

> All |ρ| < 0.04. This is expected — these are **regime characterisation tools**, not directional alpha signals. Their value is in characterising _how_ the market moved, not predicting what it will do next. Confirmed and elaborated in v2 with rank-decile tables.

---

## E) IWDS Coefficient of Variation by Window Size Bucket

| Window Size | IWDS CV (std/\|mean\|) | Reliability                         |
| ----------- | ---------------------- | ----------------------------------- |
| 10–30       | 9.72                   | ⚠ very noisy — treat as exploratory |
| 30–75       | 25.71                  | ⚠ very noisy — treat as exploratory |
| 75–150      | 16.68                  | ⚠ very noisy — treat as exploratory |
| 150–300     | 12.16                  | ⚠ very noisy — treat as exploratory |
| 300–500     | 5.69                   | ⚠ very noisy — treat as exploratory |

> CV is high because IWDS mean is near zero (market-neutral in expectation). The `⚠ noisy` overlay warning at n<30 was added to `bar_selection.rs` based on this finding.

---

## F) Regime Analysis (Low-Vol Q1 vs High-Vol Q5)

**Median values per metric per volatility quintile:**

| Metric         | Q1 median | Q5 median | Δ       |
| -------------- | --------- | --------- | ------- |
| iwds           | 0.0565    | 0.0099    | −0.0466 |
| auc            | 0.5104    | 0.5340    | +0.0236 |
| log2_ratio     | 0.0598    | 0.0453    | −0.0145 |
| conviction     | 1.0147    | 0.9939    | −0.0208 |
| absorption     | 0.4960    | 0.5016    | +0.0056 |
| climax_up_frac | 0.5224    | 0.5455    | +0.0230 |

---

## G) Edge Case Audit

| Condition                                 | Count / Value | Assessment                                                    |
| ----------------------------------------- | ------------- | ------------------------------------------------------------- |
| All-up windows (n_dn=0)                   | 0             | AUC=NaN, log2_ratio=NaN, conviction=dominant_up/NaN — correct |
| All-dn windows (n_up=0)                   | 0             | AUC=NaN, log2_ratio=NaN — correct                             |
| Small windows n<5                         | 0             | Metrics computable but noisy                                  |
| Equal-intensity AUC (avg over 200 trials) | 0.4962        | PASS (expected ≈0.5000 by symmetry)                           |
| NaN(iwds)                                 | 0             | Expected 0 (always defined)                                   |
| NaN(auc)                                  | 0             | Expected: only all-up/all-dn windows                          |
| NaN(log2_ratio)                           | 0             | Expected: only all-up/all-dn windows                          |
| NaN(conviction)                           | 0             | Expected: only all-up/all-dn or zero minority mean            |
| NaN(absorption)                           | 0             | Expected: only all-up windows (minority = dn)                 |
| NaN(climax_up_frac)                       | 0             | Expected: only n<4 windows (no top-25%)                       |

---

## Triple Barrier Calibration

| (H, TP, SL)    | Bull% | Bear% | Neutral% | Best Classifier | Best AUC | IWDS Bull Threshold |
| -------------- | ----- | ----- | -------- | --------------- | -------- | ------------------- |
| H=10 TP=2 SL=2 | 47.7% | 49.8% | 2.6%     | iwds            | 0.521    | +0.057              |
| H=10 TP=3 SL=2 | 33.0% | 55.3% | 11.7%    | iwds            | 0.520    | +0.057              |
| H=10 TP=3 SL=3 | 35.1% | 38.1% | 26.8%    | iwds            | 0.516    | +0.057              |
| H=10 TP=4 SL=2 | 23.1% | 56.5% | 20.3%    | log2_ratio      | 0.530    | +0.057              |
| H=20 TP=5 SL=3 | 26.3% | 53.7% | 20.0%    | iwds            | 0.518    | +0.057              |
| H=20 TP=5 SL=5 | 27.5% | 28.9% | 43.6%    | auc             | 0.518    | +0.057              |
| H=30 TP=5 SL=5 | 36.8% | 38.0% | 25.2%    | auc             | 0.515    | +0.057              |

> **Note on TP < SL configs**: Configs (H=10 TP=4 SL=2) with TP < SL violate the minimum risk/reward constraint (you risk more than you gain). Excluded from v2 testing per the audit policy (TP ≥ SL enforced).

**Metric AUC by classifier target** (best config: H=20 TP=5 SL=3):

| Metric         | AUC (bull vs bear) |
| -------------- | ------------------ |
| iwds           | 0.5178             |
| log2_ratio     | 0.5134             |
| auc            | 0.5086             |
| climax_up_frac | 0.5059             |
| conviction     | 0.5051             |
| absorption     | 0.5048             |

---

## OoD Robustness: Regime Sensitivity

_Q1 (low vol) n=638, Q5 (high vol) n=637_

| Metric         | KS Statistic | p-value | Assessment                           |
| -------------- | ------------ | ------- | ------------------------------------ |
| iwds           | 0.1162       | 0.0003  | 🟡 MODERATE — mild regime dependence |
| auc            | 0.2127       | 0.0000  | 🟡 MODERATE — mild regime dependence |
| log2_ratio     | 0.1429       | 0.0000  | 🟡 MODERATE — mild regime dependence |
| conviction     | 0.1041       | 0.0019  | 🟡 MODERATE — mild regime dependence |
| absorption     | 0.1010       | 0.0028  | 🟡 MODERATE — mild regime dependence |
| climax_up_frac | 0.1470       | 0.0000  | 🟡 MODERATE — mild regime dependence |

**Divergence signal stability across volatility regimes:**

| Signal                | Q1 rate | Q5 rate | Stable?                     |
| --------------------- | ------- | ------- | --------------------------- |
| climax_divergence     | 22.3%   | 38.5%   | no — regime-dependent       |
| urgency_count_diverge | 30.7%   | 45.1%   | no — regime-dependent       |
| conv_absorp_contest   | 0.0%    | 0.0%    | yes (trivially — always 0%) |

---

## Recommended Action Items (v1 → v2 mapping)

| Finding                                                           | Action Taken in v2                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `auc`, `climax_up_frac` show \|ρ\|<0.02 with all forward horizons | Retained as regime classifiers; removed from composite score weights (weight=0) |
| `conv_absorp_contest` fires 0% — logically dead                   | **Removed entirely** from `bar_selection.rs`                                    |
| IWDS CV=9.72 at n<30 — warn users                                 | `⚠ noisy` warning added to overlay header when n<30                             |
| AUC implementation verified — equal-intensity AUC=0.4962          | No code change needed                                                           |
| `conviction` ↔ `absorption` are ρ=−1.00 inverses                  | `absorption` removed; replaced by `edge` = `mean_rank_up − mean_rank_dn`        |

---

_Report generated by `audit.py` — Flowsurface ODB statistical audit v1._
_See [v2 rank audit](../v2-rank-audit/REPORT.md) for the follow-up continuous-rank analysis._

---

**GitHub**: <https://github.com/terrylica/flowsurface/blob/main/docs/audits/bar-selection-metrics/v1-threshold-audit/REPORT.md>

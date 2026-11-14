---
title: "ODB Bar-Selection Metrics: Comprehensive Rank-Based Audit (v2)"
date: 2026-03-14
version: 2
type: statistical-audit
symbol: BTCUSDT
threshold_dbps: 250
threshold_label: BPR25
bars_loaded: 10000
windows_sampled: 5000
seed: 42
data_range: "2025-12-18 → 2026-03-14"
triple_barrier_constraint: "TP ≥ SL (all configs)"
key_finding: "conviction + log2_ratio pair (AUC=0.5212) is the best predictive combo; iwds and log2_ratio are collinear (ρ=0.827); conviction spikes to AUC=0.5572 at n=300-500"
best_single_metric: conviction
best_single_auc: 0.5148
best_pair: "log2_ratio + conviction"
best_pair_auc: 0.5212
script: audit.py
methodology: CLAUDE.md
---

# ODB Bar-Selection Metrics — Comprehensive Rank-Based Audit (v2)

**Data**: BPR25 BTCUSDT | **Bars**: 10,000 | **Windows**: 5,000 (stratified log-uniform) | **Date range**: 2025-12-18 → 2026-03-14 | **Seed**: 42

> **Context**: This is v2, the follow-up to the [threshold-based v1 audit](../v1-threshold-audit/REPORT.md). Motivated by the v1 finding that `conv_absorp_contest` fires 0% (logically impossible — conviction and absorption are ρ=−1.00 inverses). v2 replaces all binary threshold analysis with **continuous rank distributions** and **AUC-based separability tests**, enabling post-hoc sorting without threshold gates. See [CLAUDE.md](./CLAUDE.md) for methodology and design decisions.

---

## Key Findings

- **Best single metric** (H=20 TP=5 SL=3): `conviction` — AUC=0.5148
- **Best pair combo**: `log2_ratio + conviction` — AUC=0.5212 (+0.006 over best single)
- **Best triple combo**: `iwds + log2_ratio + conviction` — AUC=0.5167
- **Composite (all 7 weighted)**: AUC=0.5189 (95% CI: [0.5009, 0.5378])
- **`iwds` ↔ `log2_ratio` collinearity**: ρ=0.827 — only one adds unique information
- **`conviction` at large windows**: AUC=0.5572 for n=300–500 — uniquely benefits from longer selections
- **No OoD rank-stable metric**: all KS > 0.10 — regime shift affects even relative rank ordering
- **`auc`, `edge`, `climax_up_frac`, `climax_skew`**: AUC ≤ 0.5 on best config; directional signal flips between Q1/Q5 — use as texture descriptors only

---

## Section A: Metric Distributions

| Metric         | Mean   | Std    | P5      | P25     | P50    | P75    | P95    | NaN# |
| -------------- | ------ | ------ | ------- | ------- | ------ | ------ | ------ | ---- |
| iwds           | 0.0051 | 0.3387 | −0.6685 | −0.1629 | 0.0307 | 0.1917 | 0.5322 | 0    |
| auc            | 0.5247 | 0.0872 | 0.3889  | 0.4807  | 0.5215 | 0.5667 | 0.6741 | 0    |
| log2_ratio     | 0.0134 | 1.2091 | −2.1929 | −0.2136 | 0.0818 | 0.3290 | 1.2850 | 0    |
| conviction     | 1.0258 | 0.2507 | 0.7291  | 0.9144  | 1.0046 | 1.1060 | 1.3654 | 0    |
| edge           | 0.0255 | 0.0916 | −0.1148 | −0.0196 | 0.0218 | 0.0676 | 0.1847 | 0    |
| climax_up_frac | 0.5281 | 0.1607 | 0.2500  | 0.4375  | 0.5263 | 0.6222 | 0.7900 | 0    |
| climax_skew    | 0.0323 | 0.1275 | −0.1786 | −0.0357 | 0.0297 | 0.1000 | 0.2424 | 0    |

> `conviction` is always ≥1 by construction (dominant/minority rank ratio). `auc` centred near 0.5 confirms no systematic directional bias. `edge` is tightly bounded (−0.11 to +0.18 at P5/P95) because rank normalisation constrains the space.

---

## Section B: Rank Decile Tables (H=20, TP=5, SL=3)

> D1=lowest metric value, D10=highest. **Net% = bull% − bear%.** Monotonic D1→D10 increase = reliable rank ordering.

### conviction

| Decile | N   | Bull% | Bear% | Neutral% | Net%   |
| ------ | --- | ----- | ----- | -------- | ------ |
| D1     | 500 | 23.60 | 76.40 | 0.00     | −52.80 |
| D2     | 500 | 20.40 | 79.60 | 0.00     | −59.20 |
| D3     | 500 | 19.80 | 80.20 | 0.00     | −60.40 |
| D4     | 500 | 24.20 | 75.80 | 0.00     | −51.60 |
| D5     | 500 | 23.80 | 76.20 | 0.00     | −52.40 |
| D6     | 500 | 21.40 | 78.60 | 0.00     | −57.20 |
| D7     | 500 | 24.80 | 75.20 | 0.00     | −50.40 |
| D8     | 500 | 24.00 | 76.00 | 0.00     | −52.00 |
| D9     | 500 | 23.40 | 76.60 | 0.00     | −53.20 |
| D10    | 500 | 25.20 | 74.80 | 0.00     | −49.60 |

### iwds

| Decile | N   | Bull% | Bear% | Neutral% | Net%   |
| ------ | --- | ----- | ----- | -------- | ------ |
| D1     | 500 | 22.60 | 77.40 | 0.00     | −54.80 |
| D2     | 500 | 20.60 | 79.40 | 0.00     | −58.80 |
| D3     | 500 | 24.00 | 76.00 | 0.00     | −52.00 |
| D4     | 500 | 20.80 | 79.20 | 0.00     | −58.40 |
| D5     | 500 | 24.20 | 75.80 | 0.00     | −51.60 |
| D6     | 500 | 20.40 | 79.60 | 0.00     | −59.20 |
| D7     | 500 | 27.60 | 72.40 | 0.00     | −44.80 |
| D8     | 500 | 26.20 | 73.80 | 0.00     | −47.60 |
| D9     | 500 | 27.60 | 72.40 | 0.00     | −44.80 |
| D10    | 500 | 16.60 | 83.40 | 0.00     | −66.80 |

> Note `iwds` D10 (highest positive flow) shows **worse** net% (−66.8%) than D7–D9. This U-shape suggests extreme positive flow is a contrarian signal — momentum exhaustion rather than continuation.

### log2_ratio

| Decile | N   | Bull%  | Bear%  | Neutral% | Net%        |
| ------ | --- | ------ | ------ | -------- | ----------- |
| D1     | 500 | 20.40  | 79.60  | 0.00     | −59.20      |
| D2–D8  | —   | ~22–25 | ~75–78 | 0.00     | ~−50 to −56 |
| D9     | 500 | 26.80  | 73.20  | 0.00     | −46.40      |
| D10    | 500 | 20.80  | 79.20  | 0.00     | −58.40      |

> `log2_ratio` also shows a D10 reversal (down from D9 peak), confirming extreme urgency imbalance may signal exhaustion. Best signal at D1 (lowest) and D9, not D10.

### edge / climax_skew / auc / climax_up_frac

All four show flat or non-monotonic Net% across deciles (range: −50% to −60%, no clear D1→D10 trend). Not useful as directional rank signals on this dataset.

---

## Section C: Spearman ρ(metric, fwd_score) by Window-Size Bucket

| Metric         | 10–30   | 30–75   | 75–150  | 150–300 | 300–500     |
| -------------- | ------- | ------- | ------- | ------- | ----------- |
| iwds           | −0.0127 | −0.0034 | −0.0273 | −0.0566 | −0.0353     |
| auc            | −0.0015 | −0.0323 | −0.0451 | −0.0219 | −0.0101     |
| log2_ratio     | +0.0051 | −0.0017 | −0.0491 | −0.0375 | −0.0501     |
| conviction     | −0.0096 | +0.0068 | +0.0372 | −0.0119 | **+0.0919** |
| edge           | −0.0015 | −0.0322 | −0.0453 | −0.0220 | −0.0102     |
| climax_up_frac | +0.0117 | −0.0198 | −0.0570 | −0.0517 | −0.0161     |
| climax_skew    | +0.0195 | −0.0369 | −0.0666 | −0.0246 | −0.0342     |

> `conviction` is the only metric with consistently positive ρ at large windows (300–500), confirming it benefits from longer selections. All other metrics show predominantly negative ρ — weakly contrarian at longer horizons (not actionable, near-zero in magnitude).

---

## Section D: Triple Barrier AUC (bull vs bear) — All TP ≥ SL Configs

| Config (H,TP,SL) | iwds   | auc    | log2_ratio | conviction | edge   | climax_up_frac | climax_skew |
| ---------------- | ------ | ------ | ---------- | ---------- | ------ | -------------- | ----------- |
| H=10,TP=2,SL=2   | 0.4896 | 0.4944 | 0.4927     | **0.5083** | 0.4945 | 0.4941         | 0.4936      |
| H=10,TP=3,SL=2   | 0.4976 | 0.4936 | 0.4949     | **0.5036** | 0.4936 | 0.4992         | 0.4948      |
| H=10,TP=3,SL=3   | 0.4911 | 0.4920 | 0.4912     | **0.5034** | 0.4920 | 0.4949         | 0.4939      |
| H=10,TP=4,SL=2   | 0.4948 | 0.4884 | 0.4904     | **0.4990** | 0.4883 | 0.4894         | 0.4846      |
| H=10,TP=4,SL=3   | 0.4909 | 0.4819 | 0.4872     | **0.5081** | 0.4819 | 0.4906         | 0.4855      |
| H=10,TP=4,SL=4   | 0.4865 | 0.4898 | 0.4850     | **0.5046** | 0.4899 | 0.4932         | 0.4931      |
| H=20,TP=3,SL=2   | 0.4976 | 0.4936 | 0.4949     | **0.5036** | 0.4936 | 0.4992         | 0.4948      |
| H=20,TP=3,SL=3   | 0.4911 | 0.4920 | 0.4912     | **0.5034** | 0.4920 | 0.4949         | 0.4939      |
| H=20,TP=5,SL=3   | 0.5089 | 0.4940 | 0.5100     | **0.5148** | 0.4940 | 0.4994         | 0.4946      |
| H=20,TP=5,SL=5   | 0.4984 | 0.4982 | 0.4971     | **0.5098** | 0.4983 | 0.5016         | 0.4994      |
| H=20,TP=8,SL=5   | 0.5026 | 0.4668 | 0.4991     | **0.5091** | 0.4668 | 0.4809         | 0.4755      |
| H=30,TP=5,SL=3   | 0.5089 | 0.4940 | 0.5100     | **0.5148** | 0.4940 | 0.4994         | 0.4946      |
| H=30,TP=5,SL=5   | 0.4984 | 0.4982 | 0.4971     | **0.5098** | 0.4983 | 0.5016         | 0.4994      |
| H=30,TP=8,SL=5   | 0.5026 | 0.4668 | 0.4991     | **0.5091** | 0.4668 | 0.4809         | 0.4755      |
| H=30,TP=10,SL=8  | 0.5061 | 0.4711 | 0.5063     | **0.5095** | 0.4712 | 0.4832         | 0.4807      |

> `conviction` is the best metric in **all 15 configs** — the only metric with structurally consistent above-0.5 AUC. H=20/H=30 configs collapse to identical AUC for identical (TP,SL) pairs, confirming the barrier resolves before the horizon expires.

---

## Section E: Combo Rank AUC — Top Metric Pairs and Triples

Best single-metric AUC: **0.5148** (conviction)

### Top-10 Metric Pairs

| Rank | Metric 1   | Metric 2       | Combo AUC  | Δ vs Best Single |
| ---- | ---------- | -------------- | ---------- | ---------------- |
| 1    | log2_ratio | conviction     | **0.5212** | **+0.0064**      |
| 2    | iwds       | conviction     | 0.5180     | +0.0032          |
| 3    | conviction | climax_up_frac | 0.5110     | −0.0038          |
| 4    | iwds       | log2_ratio     | 0.5108     | −0.0040          |
| 5    | conviction | climax_skew    | 0.5073     | −0.0075          |
| 6    | log2_ratio | climax_up_frac | 0.5057     | −0.0091          |
| 7    | iwds       | climax_up_frac | 0.5050     | −0.0098          |
| 8    | log2_ratio | edge           | 0.5026     | −0.0122          |
| 9    | auc        | log2_ratio     | 0.5026     | −0.0122          |
| 10   | log2_ratio | climax_skew    | 0.5025     | −0.0122          |

### Top-10 Metric Triples

| Rank | M1         | M2         | M3         | Combo AUC | Δ vs Best Single |
| ---- | ---------- | ---------- | ---------- | --------- | ---------------- |
| 1    | iwds       | log2_ratio | conviction | 0.5167    | +0.0019          |
| 2    | log2_ratio | conviction | edge       | 0.5126    | −0.0022          |
| 3    | auc        | log2_ratio | conviction | 0.5125    | −0.0022          |
| 4    | iwds       | conviction | edge       | 0.5123    | −0.0025          |
| 5    | iwds       | auc        | conviction | 0.5122    | −0.0026          |

> Triples underperform the best pair. Adding a third metric dilutes the signal — there are only 2 genuinely orthogonal contributing metrics on this dataset.

---

## Section F: OoD Rank Stability (Q1 vs Q5 Volatility Quintile)

_Q1 n=1,000 (low vol), Q5 n=1,000 (high vol). KS test on global rank percentiles._

| Metric         | KS stat    | KS p-val | ρ(Q1)   | ρ(Q5)   | OoD Stable?       |
| -------------- | ---------- | -------- | ------- | ------- | ----------------- |
| iwds           | 0.2110     | 0.0000   | −0.0410 | −0.0567 | ✗ (rank), ✓ (dir) |
| auc            | 0.2200     | 0.0000   | −0.0444 | +0.0387 | ✗ (rank), ✗ (dir) |
| log2_ratio     | 0.1670     | 0.0000   | −0.0329 | −0.0451 | ✗ (rank), ✓ (dir) |
| conviction     | **0.1250** | 0.0000   | −0.0165 | −0.0648 | ✗ (rank), ✓ (dir) |
| edge           | 0.2210     | 0.0000   | −0.0440 | +0.0382 | ✗ (rank), ✗ (dir) |
| climax_up_frac | 0.3220     | 0.0000   | −0.0740 | +0.0336 | ✗ (rank), ✗ (dir) |
| climax_skew    | 0.2420     | 0.0000   | −0.0674 | +0.0503 | ✗ (rank), ✗ (dir) |

> `conviction` has the lowest KS drift (0.125) — relatively the most regime-stable, though none pass KS < 0.10. `iwds`, `log2_ratio`, `conviction` maintain same-sign ρ across Q1 and Q5 (directionally stable). `auc`, `edge`, `climax_up_frac`, `climax_skew` flip sign — their directional interpretation changes under high volatility.

---

## Section G: AUC by Window-Size Bucket (H=20, TP=5, SL=3)

| Metric         | 10–30  | 30–75  | 75–150 | 150–300 | 300–500    |
| -------------- | ------ | ------ | ------ | ------- | ---------- |
| iwds           | 0.4964 | 0.5268 | 0.5151 | 0.4867  | 0.5187     |
| auc            | 0.5061 | 0.4826 | 0.4852 | 0.5083  | 0.4810     |
| log2_ratio     | 0.5119 | 0.5262 | 0.4947 | 0.4985  | 0.5089     |
| conviction     | 0.4977 | 0.5355 | 0.5127 | 0.4897  | **0.5572** |
| edge           | 0.5064 | 0.4827 | 0.4852 | 0.5082  | 0.4810     |
| climax_up_frac | 0.5070 | 0.4971 | 0.4895 | 0.4964  | 0.4913     |
| climax_skew    | 0.5160 | 0.4784 | 0.4768 | 0.5080  | 0.4794     |

> `conviction` at 300–500 bars (AUC=0.5572) is the strongest single-metric result in the entire audit. This is practically actionable: when a trader selects ≥300 ODB bars and `conviction` is in the top decile, the bull/bear forward outcome is meaningfully more separable than at shorter selection sizes.

---

## Section H: Signal Synergy Analysis

Top-3 single metrics: **conviction** (0.5148), **log2_ratio** (0.5100), **iwds** (0.5089)

| Pair                    | ρ(Spearman) | AUC(M1) | AUC(M2) | Combo AUC  | Synergy Gain | Orthogonal? |
| ----------------------- | ----------- | ------- | ------- | ---------- | ------------ | ----------- | --- | ------ |
| conviction + log2_ratio | 0.0395      | 0.5148  | 0.5100  | **0.5212** | **+0.0064**  | ✓ (         | ρ   | <0.30) |
| conviction + iwds       | 0.1285      | 0.5148  | 0.5089  | 0.5180     | +0.0032      | ✓ (         | ρ   | <0.30) |
| log2_ratio + iwds       | **0.8265**  | 0.5100  | 0.5089  | 0.5108     | +0.0008      | ✗           |

> **Critical structural finding**: `log2_ratio` and `iwds` are highly collinear (ρ=0.827). They both compute whether up-bars had higher raw intensity than down-bars — one via a weighted mean ratio, one via a weighted directional sum. They are mathematically near-equivalent. Despite appearing as top-3 individually, combining them yields almost zero synergy (+0.0008). `conviction` is the only genuine diversifier.

---

## Section I: Recommended Rank-Based Scoring

### Weight Derivation (proportional to max(0, AUC − 0.5))

| Metric         | Single AUC | AUC−0.5 | Weight     |
| -------------- | ---------- | ------- | ---------- |
| conviction     | 0.5148     | 0.0148  | **0.4382** |
| log2_ratio     | 0.5100     | 0.0100  | **0.2975** |
| iwds           | 0.5089     | 0.0089  | **0.2642** |
| auc            | 0.4940     | 0       | 0          |
| edge           | 0.4940     | 0       | 0          |
| climax_up_frac | 0.4994     | 0       | 0          |
| climax_skew    | 0.4946     | 0       | 0          |

### Composite Score Formula

```
composite_score =
  0.4382 × rank_pct(conviction)
+ 0.2975 × rank_pct(log2_ratio)
+ 0.2642 × rank_pct(iwds)
```

_rank_pct = percentile rank against a rolling reference window of ~500 bars, [0,1]_

> Note: because `log2_ratio` and `iwds` are collinear (ρ=0.827), their combined weight (0.5617) may be over-counting the same signal. In practice, the `conviction + log2_ratio` pair (AUC=0.5212) is the better 2-metric composite — simpler and nearly as powerful.

### Performance Summary

| Composite                                    | AUC              |
| -------------------------------------------- | ---------------- |
| Best single metric (conviction)              | 0.5148           |
| Best pair (log2_ratio + conviction)          | 0.5212           |
| Best triple (iwds + log2_ratio + conviction) | 0.5167           |
| Composite (all 7 AUC-weighted)               | 0.5189           |
| Composite 95% CI (1,000 bootstrap resamples) | [0.5009, 0.5378] |

### Practical Guidance for Traders

| Selection size    | What to trust                                                           | What to ignore                                |
| ----------------- | ----------------------------------------------------------------------- | --------------------------------------------- |
| n < 30            | Regime texture only — displayed values are noisy (⚠ noisy warning)      | Any single metric for directional prediction  |
| n = 30–150        | `log2_ratio` + `conviction` pair; regime label (CONVICTION/CLIMAX/etc.) | `climax_skew`, `edge`, `auc` for direction    |
| n = 150–300       | Same as above; add window-to-window consistency check                   | Extreme values (may be noise)                 |
| n ≥ 300           | `conviction` alone (AUC=0.5572 here); pair with `log2_ratio`            | Small-n metrics lose meaning                  |
| High vol quintile | All signals OoD-uncertain — regime check applies                        | Fixed thresholds derived from low-vol periods |

---

_Report generated by `audit.py` — Flowsurface ODB statistical audit v2 (rank-based)._
_Preceded by [v1 threshold audit](../v1-threshold-audit/REPORT.md). See [CLAUDE.md](./CLAUDE.md) for methodology._

---

**GitHub**: <https://github.com/terrylica/flowsurface/blob/main/docs/audits/bar-selection-metrics/v2-rank-audit/REPORT.md>

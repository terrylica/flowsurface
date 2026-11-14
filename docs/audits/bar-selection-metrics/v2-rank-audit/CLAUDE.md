---
title: "v2 Rank-Based Audit — Methodology & Design Decisions"
audit: v2-rank-audit
date: 2026-03-14
---

# v2 Rank-Based Metrics Audit — CLAUDE.md

**What this is**: The second statistical audit of the ODB bar-selection overlay metrics. Focuses entirely on **continuous rank distributions** rather than binary threshold triggers. Motivated by the v1 finding that `conv_absorp_contest` fired 0% (logically impossible — conviction and absorption are ρ=−1.00 mathematical inverses on the rank-normalised scale).

---

## Why Rank-Based?

v1 audited metrics as binary threshold gates (e.g., "iwds > 0.15 AND auc > 0.60 → BULL CONVICTION fires"). This hides rank structure: a metric might provide useful ordering information even when no single threshold is optimal.

v2 answers: _given the rank of a window's metric value among all windows, does that rank predict the triple-barrier outcome?_

Key difference:

- v1: `base_rate[signal_fired]` — fires or not
- v2: `AUC(metric_rank, bull_vs_bear_outcome)` — separability across the full rank distribution

---

## Metric Changes Implemented Based on v1 Findings

| Change                        | v1 State                         | v2 State                                    | Reason                                                                                                                                     |
| ----------------------------- | -------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `conv_absorp_contest`         | `bool`, fires 0%                 | **Removed entirely**                        | ρ(conviction, absorption)=−1.00 — logically impossible to have high conviction AND high absorption simultaneously on rank-normalised scale |
| `absorption` display          | `absorp: 0.49 ← minority active` | **Removed**                                 | Redundant with conviction at ρ=−1.00                                                                                                       |
| New: `edge`                   | —                                | `mean_rank_up − mean_rank_dn` ∈ [-1,+1]     | Signed, direction-aware replacement for absorption. Subsumes both conviction and absorption in one rankable scalar                         |
| `climax_divergence: bool`     | Boolean gate                     | `climax_skew: f32` ∈ [-1,+1]                | Continuous signed distance; always emitted in telemetry; rankable                                                                          |
| `urgency_count_diverge: bool` | Boolean gate                     | `urgency_split: bool` (visual trigger only) | `log2_ratio` is already the continuous rankable score; bool only triggers ⚡ row                                                           |
| Telemetry                     | `div=[CLIMAX,-,-]` flags         | `skew=-0.260 split=0` continuous scalars    | Enables post-hoc sorting/ranking of logged windows                                                                                         |
| n<30 warning                  | None                             | `⚠ noisy` in orange header                  | IWDS CV=9.72 at n<30 — audit confirmed unreliable signal                                                                                   |

---

## Script Design — `audit.py`

### Dependencies

- `clickhouse-connect` — ClickHouse HTTP connection (no pandas, cleaner)
- `numpy` — all array ops
- `scipy.stats` — Spearman ρ, KS test, rankdata

### Key Design Decisions

**Stratified log-uniform sampling** (not pure log-uniform):

- Enforces specific proportions per bucket: 30% at n=10-30, 25% at n=30-75, etc.
- Prevents over-representation of tiny windows that dominate log-uniform naturally
- Target: 5,000 windows total (vs 3,200 in v1)

**Tied-rank normalisation** (`tied_rank_norm`):

- Exact Python port of the Rust implementation in `bar_selection.rs`
- Tied bars get averaged ranks: `val = (i + j - 1) * 0.5 / (n - 1)`
- Critical for exact reproducibility of `conviction` and `edge`

**Triple-barrier forward scan**:

- TP ≥ SL constraint enforced at config level — no asymmetric risk/reward
- 16 configs tested (vs 7 in v1), covering H=10/20/30, TP/SL ratios from 1:1 to 2:1
- Horizon H=20 and H=30 collapse to identical AUC for (TP=5, SL=3) and (TP=5, SL=5) — bar sequence resolves before horizon expires

**Combo rank analysis** (new in v2):

- C(7,2)=21 pairs, C(7,3)=35 triples
- Combo score = sum of percentile ranks (equal-weight before AUC-weighting)
- NaN filled with median before percentile-ranking (don't lose windows for combos)

**OoD rank stability** (new in v2):

- KS test on **rank percentiles** (not raw values)
- Tests whether the _relative ordering_ of windows is stable across vol regimes
- Lower bar than testing raw values; none passed KS < 0.10 — regime shift affects even rank order

**Bootstrap CI** for composite AUC:

- 1,000 resamples with replacement on the valid-label subset
- Seed 42 — fully reproducible

---

## Key Findings Summary

> Full tables in [REPORT.md](./REPORT.md)

1. **`conviction` is the anchor metric** — best single AUC (0.5148), lowest KS drift (0.125), spikes to AUC=0.5572 for large windows (n=300-500). The only metric worth including in any composite.

2. **`log2_ratio` and `iwds` are collinear** (Spearman ρ=0.827). They both measure the same phenomenon: whether up-bars have higher raw intensity than down-bars. Only one contributes unique information to a composite.

3. **Best pair**: `log2_ratio + conviction` (AUC=0.5212). These two are genuinely orthogonal (ρ=0.039). The +0.006 AUC gain over the best single metric is the only statistically meaningful diversification observed.

4. **`auc`, `edge`, `climax_up_frac`, `climax_skew` do not add predictive value** on this dataset. They characterise microstructure texture (useful for reading bar anatomy) but don't rank-predict triple-barrier outcomes. Their directional signal flips between Q1 and Q5 volatility quintiles.

5. **No metric is OoD rank-stable** (all KS > 0.10). Even the relative ordering of windows shifts across volatility regimes. Any composite should carry a regime-uncertainty flag.

6. **All AUC values are near 0.5** (max 0.5572 in a specific bucket). These metrics are regime classifiers and bar anatomy descriptors, not alpha generators. The composite 95% CI lower bound is 0.5009 — edge is real but fragile.

---

## Relationship to v1

| Dimension          | v1                                    | v2                                                                            |
| ------------------ | ------------------------------------- | ----------------------------------------------------------------------------- |
| Metric output type | Binary (fired / not-fired)            | Continuous (rank scalar, always emitted)                                      |
| Signal coverage    | 6 metrics, 3 boolean divergence flags | 7 metrics, 0 boolean divergence flags (only visual triggers)                  |
| Forward outcome    | Triple-barrier AUC + Spearman ρ       | Same, plus combo AUC and rank decile tables                                   |
| OoD test           | KS on raw metric values               | KS on rank percentiles (more stringent)                                       |
| New metrics tested | —                                     | `edge` (replaces absorption), `climax_skew` (replaces climax_divergence bool) |

---

_Part of the [audits CLAUDE.md](../../CLAUDE.md) network._

---
title: "Bar-Selection Metrics — Statistical Audit Programme"
scope: docs/audits/
type: project-overview-claude-md
---

# Audit Programme — CLAUDE.md

**Purpose**: Every statistical claim in the bar-range selection overlay (`src/chart/kline/bar_selection.rs`) must be backed by a reproducible, versioned audit against real ClickHouse data. This directory is the permanent record.

---

## Directory Layout

```
docs/audits/
└── bar-selection-metrics/
    ├── v1-threshold-audit/     Audit v1 — binary threshold signals (March 2026)
    │   ├── audit.py            Script: pandas + scipy + rich, ClickHouse HTTP
    │   ├── pyproject.toml      Pinned Python deps for full reproducibility
    │   └── REPORT.md           Findings with YAML frontmatter
    └── v2-rank-audit/          Audit v2 — continuous rank-based metrics (March 2026)
        ├── audit.py            Script: clickhouse-connect + numpy + scipy, no pandas
        ├── CLAUDE.md           Nested CLAUDE.md — methodology, design decisions
        └── REPORT.md           Findings with YAML frontmatter + GitHub URL
```

---

## Quick Reference — Audit Versions

| Version                                                                    | Focus                                                                      | N windows | Key Finding                                                                                                                                                               |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [v1 — threshold audit](bar-selection-metrics/v1-threshold-audit/REPORT.md) | Binary signal base rates, ρ matrix, triple-barrier calibration, OoD KS     | 3,200     | `conviction ↔ absorption` are ρ=−1.00 inverses; `conv_absorp_contest` fires 0%; all divergence base rates healthy (28–41%)                                                |
| [v2 — rank audit](bar-selection-metrics/v2-rank-audit/REPORT.md)           | Continuous rank distributions, combo AUC, window-size interaction, synergy | 5,000     | `conviction + log2_ratio` best pair (AUC=0.5212); `iwds` and `log2_ratio` nearly collinear (ρ=0.83 — only one adds value); `conviction` spikes to AUC=0.5572 at n=300–500 |

---

## What These Metrics Measure

All metrics are computed from `trade_intensity = individual_trade_count / bar_duration_seconds` — the Engle–Russell (1998) ACD measure of market urgency. They characterise the **microstructure texture** of a selected bar range, not price direction per se.

| Metric           | Formula                                       | Interpretation                                                                                    |
| ---------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `iwds`           | `Σ(intensity_i × ±1) / Σ(intensity_i)`        | Intensity-weighted directional score [-1, +1]. +1 = all urgency on up-bars                        |
| `auc`            | Mann-Whitney U / (n_up × n_dn)                | P(random up-bar > random dn-bar in intensity). 0.5 = no structural edge                           |
| `log2_ratio`     | `log₂(mean_intensity_up / mean_intensity_dn)` | Logarithmic ratio of up vs down urgency. +1 = 2× advantage on up side                             |
| `conviction`     | `mean_rank_dominant / mean_rank_minority`     | How much the dominant side outranks the minority in rank-normalised intensity. Always ≥ 1         |
| `edge`           | `mean_rank_up − mean_rank_dn`                 | Signed rank-normalised intensity edge [-1, +1]. Replaces `absorption` (ρ=−1.00 with `conviction`) |
| `climax_up_frac` | `count(up in top-25%) / count(top-25%)`       | Fraction of highest-intensity moments that are up-bars. Tail concentration signal                 |
| `climax_skew`    | `climax_up_frac − (n_up / n)`                 | How much the climax fraction deviates from overall count fraction. Negative = divergence signal   |

---

## Reproducibility

### Prerequisites

- SSH tunnel active: `mise run tunnel:start` (routes `localhost:18123 → <clickhouse-host>:8123`)
- Data: `opendeviationbar_cache.open_deviation_bars` (BTCUSDT, threshold=250)

### Run v1 (threshold audit)

```bash
cd docs/audits/bar-selection-metrics/v1-threshold-audit
uv run --python 3.13 audit.py
# Report: /tmp/flowsurface-audit/AUDIT_REPORT.md
```

### Run v2 (rank audit)

```bash
cd docs/audits/bar-selection-metrics/v2-rank-audit
uv run --python 3.13 --with clickhouse-connect --with numpy --with scipy audit.py
# Report: /tmp/flowsurface-audit2/AUDIT2_REPORT.md
```

---

## Audit Policy

- **When to re-audit**: after any change to metric formulas in `bar_selection.rs`, after significant new data (>10k new bars), or after major market regime shifts.
- **Versioning**: create a new `vN-*` subdirectory per audit run; never overwrite prior reports.
- **TP ≥ SL constraint**: all triple-barrier configs must satisfy TP ≥ SL to ensure a minimum risk/reward ratio. Configs violating this are excluded (see v2 audit).
- **Rank not threshold**: from v2 onwards, report continuous rank distributions and AUC curves, not binary threshold firing rates. Thresholds hide rank structure.

---

## Code Reference

| File                                 | Purpose                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `src/chart/kline/bar_selection.rs`   | Metric computation + rendering. All formulas here are the authoritative source      |
| `data/src/aggr/ticks.rs`             | `TickAggr` — bar storage. `trade_intensity` lives in `OdbMicrostructure`            |
| `exchange/src/adapter/clickhouse.rs` | ClickHouse fetch: `individual_trade_count`, `close_time_ms`, `open_time_ms` columns |

---

_Hub CLAUDE.md for `docs/audits/`. Part of the [CLAUDE.md network](/CLAUDE.md)._

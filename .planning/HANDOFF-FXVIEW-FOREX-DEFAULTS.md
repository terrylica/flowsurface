# Handoff — flowsurface forex default threshold + readiness status

**From:** `~/eon/mql5/` session (fxview-sidecar / forex_bars producer)
**To:** `~/fork-tools/flowsurface/` session (consumer)
**Date:** 2026-04-14
**Status after 5-team audit:** consumer contract works, but one UX blocker

---

## TL;DR

`fxview_cache.forex_bars` is schema-complete and live-producing on bigblack. Your existing consumer code in `exchange/src/adapter/clickhouse.rs` already works — types, column names, and µs time unit match. **The only blocker on your side is the hardcoded default threshold of 250 dbps in `src/chart/kline/mod.rs:1639`.** Forex bars only exist at thresholds 5/10/25 dbps, so any user selecting EURUSD or XAUUSD at default sees zero bars.

---

## What to fix in flowsurface

### 1. Symbol-aware default threshold (REQUIRED)

**File:** `src/chart/kline/mod.rs` around line 1639
**Current:**

```rust
const DEFAULT_THRESHOLD_DBPS: u32 = 250; // or wherever the literal lives
```

**Change to:** symbol-conditional default.

- Forex (EURUSD, XAUUSD, and any future FXView pairs): **5 dbps**
- Crypto (everything else, existing behavior): **250 dbps**

Reuse the existing crypto-vs-forex classifier if you already have one (Team B's audit confirmed flowsurface dispatches via `!ends_with("USDT") && !ends_with("BUSD")` in the ClickHouse adapter — that same predicate can select the default). If not, add a small helper:

```rust
fn default_threshold_dbps(symbol: &str) -> u32 {
    const FOREX: &[&str] = &["EURUSD", "XAUUSD"];
    if FOREX.contains(&symbol) { 5 } else { 250 }
}
```

### 2. Extend `validate_schema()` to cover forex_bars (RECOMMENDED)

Currently `validate_schema()` only checks `opendeviationbar_cache.open_deviation_bars`. If our schema drifts (we add/remove columns), the crypto-side validator fires but the forex side silently serde-defaults and produces wrong-but-valid data. Add a parallel check against `fxview_cache.forex_bars` for the 8 columns you actually read: `open_time_us, close_time_us, open, high, low, close, quote_count, duration_us, symbol, threshold_decimal_bps, ouroboros_mode`.

### 3. No other changes needed

- Types match 1:1 — no deser changes.
- Time unit is µs, your adapter converts at 5 sites via `/1000`. Keep doing that.
- Symbol strings are bare (`EURUSD`, not `FXVIEW_EURUSD`) — matches your current assumption.
- `quote_count AS individual_trade_count` aliasing you already do — keep.

---

## Our-side state (informational)

| Component              | Status                     | Notes                                                    |
| ---------------------- | -------------------------- | -------------------------------------------------------- |
| Schema                 | ✅ production-grade        | 66 cols, 100% COMMENT coverage, types/codecs fixed       |
| Live pipeline          | ✅ running on bigblack     | sidecar writing 5.8 ticks/s in NY session                |
| Historical backfill    | 🟡 triggered just now      | EA replaying 2018 → today; ETA several hours             |
| Sessions enrichment    | 🟡 service being installed | `fxview-sessions-enricher` — populates 5 session columns |
| Clean build provenance | 🟡 rebuild in progress     | version will drop `-dirty` suffix soon                   |

Contract columns flowsurface consumes (stable — treat as frozen):

```
open_time_us             Int64    µs since epoch
close_time_us            Int64    µs since epoch
open, high, low, close   Float64  price
quote_count              UInt32   tick update count (your alias: individual_trade_count)
duration_us              Int64    close - open, µs
symbol                   LowCardinality(String)
threshold_decimal_bps    UInt16   5, 10, or 25 for forex
ouroboros_mode           LowCardinality(String)
```

Any rename, retype, or unit change will go through an explicit handoff like this one — we will not silently break.

---

## Out of scope for flowsurface

- You do NOT need to query `session_*`, `lookback_*`, `intra_*`, Welford moments, gap detection columns, provenance columns, or any of the 50+ other columns we emit. They're there for downstream quant research, not charting.
- You do NOT need to handle forex backfill / catchup / SSE. Forex streams live-only on your side — the `fxview-backfiller` binary on bigblack handles historical regeneration into ClickHouse.

---

## Contact

Any schema question → open an issue on `terrylica/mql5` or ping the parallel session. The `.planning/agent-outputs/STATUS-{A..E}-*.md` files in `~/eon/mql5/` have the full 5-team audit raw data.

---

## Addendum (2026-04-15) — news_events companion table now available

New table `fxview_cache.news_events` populated by THREE independent sources for redundancy + cross-validation. Each row carries a `source` column (`mt5` | `forexfactory` | `gdelt`) so consumers can filter or validate by counting matched rows across sources.

### Current row counts

| Source | Rows | Span |
|---|---:|---|
| `mt5` | 194,089 | 2010-2026 (scheduled macro releases, second-precision UTC, ~30 countries) |
| `forexfactory` | 107 rolling + grows via 30-min systemd timer | current week (forex-focused) |
| `gdelt` | 8,687,458 | 2018-2026 (geopolitics + OFAC + CB surprises, 15-min granularity) |

### Recommended consumption pattern (annotate bars with news)

```sql
SELECT b.symbol, b.open_time_us, b.close_time_us, b.open, b.high, b.low, b.close,
       n.event_name, n.source, n.impact_level
FROM fxview_cache.forex_bars b
LEFT JOIN fxview_cache.news_events n
  ON n.event_time_us BETWEEN b.close_time_us - 900000000 AND b.close_time_us + 900000000  -- ±15 min
  AND (n.country = 'US' OR (b.symbol = 'EURUSD' AND n.country = 'EUR'))
WHERE b.symbol = 'EURUSD' AND b.threshold_decimal_bps = 5
  AND b.close_time_us BETWEEN {from_us} AND {to_us}
```

### Empirical correlation (see `.planning/agent-outputs/NEWS-BAR-CORRELATION.md`)

- 62% of extreme-range bars cluster at 21-22 UTC — forex broker rollover, not news-driven.
- 100% of non-rollover extreme bars match a GDELT geopolitical event within ±5 min.
- MT5 scheduled-release matches are sensitive to the join window: ±5min misses, ±15min likely hits. Tune the window empirically.

### Schema contract (frozen)

Columns consumers should use:
- `event_time_us` Int64 — UTC µs (JOIN key)
- `country` LowCardinality(String) — ISO-alpha-3 or central-bank issuer
- `event_name` LowCardinality(String) — human-readable
- `impact_level` UInt8 — 0 unrated, 1 low, 2 med, 3 high
- `source` LowCardinality(String) — provenance for filtering/cross-validation

All 12 columns are COMMENTed in CH per project SSoT rule. `SHOW CREATE TABLE fxview_cache.news_events` on bigblack for the authoritative schema.

### Out of scope

flowsurface is NOT expected to consume news_events in the initial integration — it's optional chart annotation when the UI is ready. The table stands up now so the data is ready when you are.

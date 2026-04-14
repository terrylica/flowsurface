# Handover from the mql5 session (2026-04-14): FXView forex pipeline overhaul

Written by the Claude session working in `~/eon/mql5/`. Scope: the upstream
pipeline that feeds forex ODB bars you're trying to render.

**TL;DR — three things you need to know right now:**

1. Your EURUSD@25 "42.5 dbps overshoot" was **real but misdiagnosed**. It was
   NOT a sidecar-version regression. It was **post-shutdown tick replay** —
   fixed at the DLL level today. Issue `terrylica/opendeviationbar-py#364`
   should be closed or redirected.
2. **Forex data has moved to a new database and table.** Your queries that
   read `opendeviationbar_cache.open_deviation_bars` for EURUSD/XAUUSD
   **will return zero rows from now on.** Point at `fxview_cache.forex_bars`
   instead.
3. The new forex schema is **quote-native** — separate bid/ask/mid OHLC
   columns, spread stats, forex-specific lag telemetry. Crypto in the old
   table is unchanged.

---

## What the rendering glitch actually was

You observed bars with 42.5 dbps avg range on a 25 dbps threshold, but only
on Apr 14. Older bars (Apr 6-13) looked correct at ~3.7 dbps range.

The real root cause chain:

- At some point today, MT5 was shut down and restarted.
- The MT5 EA `TickCollector.mq5` uses a watermark-resume pattern:
  `CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, last_time_msc, 0)`.
- On restart, that call replays every historical tick since the last
  Parquet footer — potentially hours of stale ticks.
- Each replayed tick was unconditionally published to
  `/dev/shm/tick_ring_{SYMBOL}`, and the downstream sidecar saw stale
  timestamps as "fresh" sequence increments.
- Bar construction on stale-tick-intervals produces pathological ranges
  because a "bar" could span hours of real time compressed into
  milliseconds of replay, all within the threshold window accounting.

This has been fixed at three levels today, in commits that shipped as
`mql5` releases `1.12.0` → `1.12.2`:

| Layer             | Commit    | Fix                                                                                                                                                                                                                           |
| ----------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tick-writer` DLL | `1d5cf6d` | Monotonic gate in `TickRing::publish` seeded from last-slot `time_msc` on open, so post-restart replays are rejected at the producer                                                                                          |
| `tick-writer` DLL | `d017908` | Initial wall-clock gate (superseded by monotonic gate after discovery of broker-time offset)                                                                                                                                  |
| MQL5 EA           | `6d95255` | Two interlocking fixes: broker-time→UTC conversion before the DLL, and first-run watermark = `TimeTradeServer() * 1000` instead of 0 (which would otherwise replay MT5's entire local tick cache — 8 years for FXView EURUSD) |

So when issue #364 references "13.73.0+forex-streaming" as the culprit,
that's not the real cause. That version string comes from the Python
sidecar that USED to write forex to `opendeviationbar_cache`. The bad
bars were an artifact of the producer (the DLL + EA), not the consumer.

**The entire Python forex-streaming path has been retired.** See below.

---

## Architectural change you need to know about

As of tonight, forex operations are fully separated from crypto:

| Dimension               | Crypto                                          | FXView forex                                            |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| Producer                | Binance WebSocket                               | MT5 (Wine) → `tick_writer.dll` → `/dev/shm/tick_ring_*` |
| Sidecar                 | `opendeviationbar-py/scripts/` (Python)         | `~/eon/mql5/tools/fxview-sidecar/` (Rust)               |
| ClickHouse DB           | `opendeviationbar_cache`                        | `fxview_cache`                                          |
| Table                   | `open_deviation_bars`                           | `forex_bars`                                            |
| systemd unit (bigblack) | `opendeviationbar-*.service`                    | `fxview-sidecar.service`                                |
| Schema style            | Trade-centric (aggTrade IDs, buyer_maker, etc.) | Quote-native (bid/ask OHLC, spread stats)               |

Your flowsurface queries currently read `opendeviationbar_cache.open_deviation_bars`
filtered by `symbol IN ('EURUSD','XAUUSD')`. As of tonight, that table has
**zero forex rows** and **no new forex rows will be written there**. Switch
all forex queries to `fxview_cache.forex_bars`.

---

## Schema of the new forex table

Full DDL lives in `~/eon/mql5/tools/fxview-sidecar/schema.sql`. Summary
of columns relevant to rendering:

### Keys

- `symbol` — LowCardinality(String) — `EURUSD`, `XAUUSD`
- `threshold_decimal_bps` — UInt32 — `5`, `10`, `25`, `50` (1 dbps = 0.001%)
- `open_time_us` / `close_time_us` — Int64 microseconds **UTC** (not broker-time anymore)

### Mid-price OHLC (the price you'll most likely render)

- `open` / `high` / `low` / `close` — all `(bid+ask)/2` at the relevant tick

### Bid-side OHLC (for spread-aware rendering)

- `bid_open` / `bid_high` / `bid_low` / `bid_close`

### Ask-side OHLC

- `ask_open` / `ask_high` / `ask_low` / `ask_close`

### Activity

- `quote_count` — UInt32 — number of quote-ticks in the bar (not trades —
  FXView quotes are bid/ask updates, no executed trades are reported)
- `duration_us` — Int64 microseconds

### Spread stats (all in price, not dbps)

- `spread_open` / `spread_high` / `spread_low` / `spread_close`
- `spread_mean` — simple arithmetic mean over quotes in the bar

### Forex session flags

Currently populated as `0` / `""` (placeholder). The MVP sidecar skipped
chrono-tz + holiday calendar integration; will be populated by a follow-up.
If your UI wants live session badges today, compute them client-side from
`close_time_us` UTC. Columns exist so the schema is stable:

- `session_sydney` / `session_tokyo` / `session_london` / `session_newyork`
  — UInt8 flags (0 = outside, 1 = inside)
- `session_label` — LowCardinality(String) — e.g. `london_ny`, `tokyo_sydney`

### Ouroboros

- `ouroboros_mode` — LowCardinality(String) — always `week` for forex
  (weekend cutover; weekly open Sunday 21:00 UTC approx). Part of `ORDER BY`.

### NEW — per-bar lag telemetry (useful for rendering quality badges!)

- `perbar_fx_quote_close_ingest_lag_us` — microseconds between bar close
  and when the sidecar first observed it from the ring. Proxy for
  broker→MT5→DLL→ring latency. Normal: sub-second. Spikes: upstream stall.
- `perbar_fx_quote_max_gap_us` — longest inter-tick silence within the
  bar. Detects illiquid periods. A 60-second bar with a 45-second internal
  silence is structurally different from one with 100 ms gaps — you could
  render a "quiet bar" indicator in flowsurface.
- `perbar_fx_quote_close_write_lag_us` — microseconds between bar close
  and ClickHouse write. Complements the ingest lag.

### Provenance

- `sidecar_version` — LowCardinality(String) — comes from Cargo pkg
  version; current value `1.12.2`.
- `broker_utc_offset_sec` — Int32 — broker's UTC offset at write time (e.g.
  `10800` for EEST). Diagnostic trail if MT5 ever misreports tz again.
- `computed_at` — DateTime64(3, 'UTC') — ReplacingMergeTree version column.

### Engine + keys

```sql
ENGINE = ReplacingMergeTree(computed_at)
PARTITION BY (symbol, threshold_decimal_bps)
ORDER BY (symbol, threshold_decimal_bps, ouroboros_mode, open_time_us)
```

---

## Example migration of your queries

### Before (legacy path)

```sql
SELECT open, high, low, close, close_time_us
FROM opendeviationbar_cache.open_deviation_bars
WHERE symbol = 'EURUSD' AND threshold_decimal_bps = 25
ORDER BY close_time_us DESC LIMIT 500
```

### After (new path)

```sql
SELECT open, high, low, close, close_time_us,
       bid_close, ask_close, spread_mean,
       perbar_fx_quote_max_gap_us
FROM fxview_cache.forex_bars
WHERE symbol = 'EURUSD' AND threshold_decimal_bps = 25
ORDER BY close_time_us DESC LIMIT 500
```

Note the schema surface has grown: you can now render bid/ask bands,
spread widths, and a tick-drought indicator if you want.

---

## Data lifecycle caveats

- **All pre-2026-04-14 forex data is gone.** Clean slate was executed per
  user directive tonight. If you had any queries or caches pinned to
  older bars, they no longer exist anywhere — not in the old table
  (partitions dropped), not in the new table (started fresh).
- **XAUUSD tick feed pauses ~21:00-22:00 UTC daily** (FXView's gold
  settlement rollover window). The sidecar is running but `quote_count`
  will be zero for bars trying to form during that window. Not a bug;
  don't render it as a pipeline failure.
- **Week ouroboros boundary is not yet enforced** by the MVP sidecar.
  Weekend bars may span from Friday close to Monday open without a
  forced close. Follow-up item tracked in
  `~/eon/mql5/tools/fxview-sidecar/README.md`.

---

## On issue #364

The filed issue blames the "13.73.0+forex-streaming" version downgrade.
That's not accurate — the real cause was post-shutdown replay in the
producer DLL, and it's fixed today (commits `d017908`, `1d5cf6d`,
`6d95255`). The `forex-streaming` sidecar suffix is a red herring; the
Python path has since been retired entirely.

Suggested action: **add a comment on `terrylica/opendeviationbar-py#364`
with the real root cause, and close it** (or reopen as a new clean issue
against the mql5 repo if you want to track the forex-separation work).
The `opendeviationbar-py` repo is not where forex bugs are fixed anymore.

---

## Files to read for full context

In `~/eon/mql5/`:

- `tools/fxview-sidecar/README.md` — operator guide (build, deploy, env config)
- `tools/fxview-sidecar/schema.sql` — authoritative DDL with COMMENT on every column
- `tools/fxview-sidecar/src/bars.rs` — streaming bar-construction algorithm (~200 lines)
- `mql5_ea/TickCollector.mq5` — the MT5 EA with the broker→UTC conversion
- `crates/tick-writer/src/ring.rs` — the ring producer with the monotonic gate
- `crates/tick-writer/src/writer.rs` — DLL layer (writes Parquet + WAL + ring)

In `~/eon/opendeviationbar-patterns/.planning/`:

- `FOLLOWUP-OPENDEVIATIONBAR-PY-FOREX-DELETION.md` — spec for deleting the
  dead forex code from `opendeviationbar-py` (pending, not your concern)
- `HANDOFF-ZIGZAG-V4.md` — earlier handover (zigzag research context)

---

## What you can help with / where we can collaborate

From a rendering perspective, the new `fxview_cache.forex_bars` table
lets you do things you couldn't easily do in the trade-centric schema:

1. **Render bid/ask bands around each bar** using the separate bid/ask
   OHLC columns. Useful for visualizing spread widening during volatile
   periods.
2. **Tick-drought indicator** from `perbar_fx_quote_max_gap_us`. A simple
   overlay: if the worst intra-bar gap exceeds some threshold (say 30s
   for liquid forex), render a badge.
3. **Ingest-lag health badge** from `perbar_fx_quote_close_ingest_lag_us`.
   Sub-second = green, 1-5s = yellow, >5s = red. Helps operators spot
   broker-feed issues instantly.
4. **Spread visualization** via `spread_mean` / `spread_high` / `spread_low`.
   A gold chart showing when the spread ballooned during session
   transitions would be a unique flowsurface feature.

If any of this is interesting and the schema needs adjustment for your
rendering needs, propose the schema change — we can add columns without
clean-slating again (the current DB and schema are designed to evolve via
`ALTER TABLE ADD COLUMN (Nullable)`).

---

_Written 2026-04-14 by the mql5 session. Release tags `1.12.0`-`1.12.2`
correspond to the three fixes in this handover._

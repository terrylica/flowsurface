# Handoff to opendeviationbar-py — crypto cache data lacuna

**From:** `~/fork-tools/flowsurface/` autoloop session (consumer)
**To:** `~/eon/opendeviationbar-py/` team (producer of `opendeviationbar_cache.open_deviation_bars`)
**Date:** 2026-05-01
**Trigger:** User reported "no earlier data for ODB at 500 dbps" on BTCUSDT chart;
forensic probe confirmed two distinct producer-side incidents.

---

## ⚠️ STATUS: Diagnosis correct, remediation asks superseded by producer intent

After the producer team reviewed this handoff (their session
`a87c996b-1f0a-4a2b-b07f-b473660ed38c`), they confirmed the data shape
described below is **intentional clean-slate POC scope**, not a failure
condition. They explicitly chose **hold — no reply doc, no scope changes**.

Specifically:

- **BTCUSDT/ETHUSDT 33-day window** is the result of `repair_direct_parquet.py
--recent-days 30` run during today's iter-11 deploy (v13.75.0). Older
  history was wiped during prior clean-slate POC rebuilds — by design.
- **14 "frozen" symbols** have `enabled = false` in `symbols.toml`. Only
  BTCUSDT + ETHUSDT have `enabled = true`. The 14 symbols' "stale" bars are
  legacy data from before the crypto-narrowing.
- **86-hour dark spell at 2026-04-12** was incident #363; root regression
  fixed today in commit `d0d4ffe5` (v13.75.0). Our consumer is now on
  the same crate version (Round 5 of the autoloop bumped 13.70.3 → 13.75.0).
- **Producer-side gate**: `OPENDEVIATIONBAR_KINTSUGI_GENESIS_ENABLED=false`
  is the explicit env that "prevents undoing clean-slate POC rebuilds."
  Will be flipped to `true` when ready for full-history backfill.

**Net for flowsurface**: my original "remediation asks" below would actively
undo the producer's deliberate POC state. They should be read as
"observations of state" rather than "asks." The full diagnosis is left
intact below as a forensic record; the asks at the end are obsolete.

---

## TL;DR (original — observation only, not action)

`opendeviationbar_cache.open_deviation_bars` is in a degraded state with two
incidents conflated:

1. **All 14 historic-data symbols frozen since 2026-04-11/12/13.** Last bars
   landed within a 48-hour window across symbols spanning AAVEUSDT through
   XRPUSDT. The main ingestion pipeline died ~3 weeks ago and nobody noticed.

2. **BTCUSDT + ETHUSDT data deleted.** They went from ~10M+ bars each (matching
   peer symbols like ADAUSDT 10.3M, XRPUSDT 10.4M) to 13K/22K respectively.
   Then re-ingestion was re-enabled for _only those two_ symbols starting
   2026-03-28. So they have _fresh_ data but _no history_.

Net effect on flowsurface users: BTCUSDT and ETHUSDT charts show only 33 days
of bars instead of the multi-year history users expect; all other symbols are
frozen 3 weeks in the past.

---

## Evidence

### Cross-symbol bar counts (CH query 2026-05-01 02:33 UTC)

| Symbol      |       Bars | Earliest       | Latest     | Status                             |
| ----------- | ---------: | -------------- | ---------- | ---------------------------------- |
| **BTCUSDT** | **13,307** | **2026-03-28** | 2026-05-01 | wiped + partial re-ingest          |
| **ETHUSDT** | **22,030** | **2026-03-28** | 2026-05-01 | wiped + partial re-ingest          |
| BNBUSDT     |  8,809,852 | 2018-01-16     | 2026-04-11 | frozen ~3w ago                     |
| LTCUSDT     |  1,861,561 | 2018-01-16     | 2026-04-12 | frozen ~3w ago                     |
| ADAUSDT     | 10,302,676 | 2018-04-17     | 2026-04-12 | frozen ~3w ago                     |
| XRPUSDT     | 10,420,515 | 2018-05-04     | 2026-04-11 | frozen ~3w ago                     |
| DOGEUSDT    | 16,751,290 | 2019-07-05     | 2026-04-12 | frozen ~3w ago                     |
| LINKUSDT    |  7,911,352 | 2019-01-16     | 2026-04-13 | frozen ~3w ago                     |
| BCHUSDT     |  5,711,842 | 2019-11-28     | 2026-04-13 | frozen ~3w ago                     |
| SOLUSDT     |  9,392,947 | 2020-08-11     | 2026-04-11 | frozen ~3w ago                     |
| AVAXUSDT    |  8,831,384 | 2020-09-22     | 2026-04-12 | frozen ~3w ago                     |
| UNIUSDT     |  8,233,056 | 2020-09-17     | 2026-04-11 | frozen ~3w ago                     |
| AAVEUSDT    |  8,286,114 | 2020-10-15     | 2026-04-12 | frozen ~3w ago                     |
| FILUSDT     |  7,369,072 | 2020-10-15     | 2026-04-12 | frozen ~3w ago                     |
| NEARUSDT    |    165,282 | 2020-10-14     | 2026-04-12 | frozen ~3w ago                     |
| SUIUSDT     |  3,631,844 | 2023-05-03     | 2026-04-11 | frozen ~3w ago                     |
| **TOTAL**   | **107.7M** |                |            | (was 171M per 2026-04-11 snapshot) |

The 63M-row deficit between snapshots aligns with the BTCUSDT + ETHUSDT
wipes (those two were ~10M each at full backfill; combined deletion explains
~20M; the rest is unclear — possibly other symbols had selective row-level
deletion not captured here).

### Single-incident signature

The 14 frozen symbols' last bar timestamps cluster tightly:

```
2026-04-11: BNBUSDT, XRPUSDT, SOLUSDT, UNIUSDT, SUIUSDT
2026-04-12: ADAUSDT, DOGEUSDT, AAVEUSDT, AVAXUSDT, LTCUSDT, NEARUSDT, FILUSDT
2026-04-13: BCHUSDT, LINKUSDT
```

That's a 48-hour cluster — characteristic of a single pipeline death, not
14 independent failures.

### data_freshness table reveals operational state

```
opendeviationbar_cache.data_freshness:
  BTCUSDT, ETHUSDT (all 4 thresholds): updated_at as recent as 2026-05-01 02:33 UTC ✓
  (Other symbols: not present in data_freshness — never re-registered after wipe?)
```

The presence of BTCUSDT/ETHUSDT in data_freshness with active updates,
plus absence of the other 14 symbols, suggests a manual operation
re-enabled tracking for only the two flagship symbols.

---

## What we're asking for

1. **Full backfill of BTCUSDT and ETHUSDT** to restore the multi-year
   history that was deleted. Pre-wipe these had ~10M+ bars each going
   back to 2018 (BTCUSDT) and ~2017 (ETHUSDT). The 33-day current window
   is unusable for any historical analysis.

2. **Resume ingestion of the 14 frozen symbols** (or document that they're
   intentionally archived — but if so, they should be removed from
   active discovery via `init_odb_symbols()` so they don't appear in
   the ticker picker showing stale data).

3. **Post-incident root-cause** — what stopped ingestion ~2026-04-11/12/13?
   Single pipeline death suggests one upstream failure (Binance API
   credentials, opendeviationbar-py service crash, ClickHouse partition
   issue, disk pressure, etc.). Document so it doesn't recur silently.

---

## Flowsurface side: zero changes needed

The consumer correctly:

- Discovers all 16 symbols at startup via `init_odb_symbols()`
- Fetches 100% of available bars for each symbol/threshold combination
- Renders the chart faithfully (no interpolation, no concealment of gaps)

Users are correctly seeing what's in CH — which is the deficit the producer
needs to remediate.

The user's observation "no earlier data for ODB at 500 dbps" was a precise
diagnosis: ALL BTCUSDT thresholds (100/250/500/750) start at 2026-03-28.
The BPR50 view just happened to be the first one they noticed it on.

---

## Suggested action sequence (for opendeviationbar-py team)

```bash
# 1. Confirm the full state on bigblack
ssh bigblack 'clickhouse-client -q "
SELECT symbol, count() AS bars,
       toDate(fromUnixTimestamp64Micro(min(close_time_us))) AS earliest,
       toDate(fromUnixTimestamp64Micro(max(close_time_us))) AS latest
FROM opendeviationbar_cache.open_deviation_bars
GROUP BY symbol ORDER BY bars
"'

# 2. Check ingestion service state
ssh bigblack 'systemctl --user list-units --all | grep -iE "opendeviationbar|kintsugi"'

# 3. Check journal for what failed ~2026-04-11/12/13
ssh bigblack 'journalctl --user --since "2026-04-10" --until "2026-04-14" --no-pager | grep -iE "error|panic|fatal" | tail -50'

# 4. Verify Binance credentials still valid
ssh bigblack 'systemctl --user status opendeviationbar-stream.service 2>/dev/null'

# 5. Re-trigger backfill for BTCUSDT/ETHUSDT to recover multi-year history
# (specific command depends on opendeviationbar-py orchestration; check
#  scripts/orchestrate.py or equivalent backfill entry point)
```

---

_Written 2026-05-01 by the flowsurface autoloop session after user-pushed
re-investigation of "no earlier data for ODB at 500 dbps." First analysis
incorrectly suggested a partial-backfill of 4 symbols; the user's intuition
that "the data source must have some problems" was sharper, and the deeper
probe confirmed: it's actually a 14-symbol freeze + 2-symbol wipe._

# Crypto data inventory — what's actually on bigblack right now

**Purpose:** Forensic answer to "look deep underneath the ClickHouse database
and look at what's really available and why we cannot get them."

**Date:** 2026-05-01 (probe captured at ~03:30 UTC)
**Host:** bigblack
**Scope:** Crypto-related data sources for opendeviationbar bar generation

---

## TL;DR

Multi-year crypto data IS available across **4 layers**, but only the
top-most (active in ClickHouse) is reachable from Flowsurface. The other
3 layers are gated by **one env var**: `OPENDEVIATIONBAR_KINTSUGI_GENESIS_ENABLED=false`.

| Layer                                | Magnitude                | Reachability      | Time-sensitivity                    |
| ------------------------------------ | ------------------------ | ----------------- | ----------------------------------- |
| 1. Active CH bars                    | 107.7M rows / 13.77 GiB  | ✅ Reachable      | Permanent                           |
| 2. Inactive CH parts (dropped today) | 55.10 GiB / 4,152 parts  | ⚠ Pending GC      | **Recoverable for hours, not days** |
| 3. Binance source archives           | ~10 years × 7-10 symbols | ✅ Present, gated | Permanent (raw zips)                |
| 4. opendeviationbar processing cache | 113 GB                   | ✅ Present, gated | Permanent (regeneratable)           |

---

## Layer 1 — Active in ClickHouse (what Flowsurface sees)

```
opendeviationbar_cache.open_deviation_bars: 107,714,138 rows / 13.77 GiB
```

Per-symbol distribution (across all thresholds):

| Symbol   |       Bars | Earliest   | Latest     | Status                           |
| -------- | ---------: | ---------- | ---------- | -------------------------------- |
| BTCUSDT  |     13,307 | 2026-03-28 | 2026-05-01 | Live (POC scope)                 |
| ETHUSDT  |     22,030 | 2026-03-28 | 2026-05-01 | Live (POC scope)                 |
| DOGEUSDT | 16,751,290 | 2019-07-05 | 2026-04-12 | Frozen (legacy, `enabled=false`) |
| XRPUSDT  | 10,420,515 | 2018-05-04 | 2026-04-11 | Frozen (legacy)                  |
| ADAUSDT  | 10,302,676 | 2018-04-17 | 2026-04-12 | Frozen (legacy)                  |
| SOLUSDT  |  9,392,947 | 2020-08-11 | 2026-04-11 | Frozen (legacy)                  |
| BNBUSDT  |  8,809,852 | 2018-01-16 | 2026-04-11 | Frozen (legacy)                  |
| AVAXUSDT |  8,831,384 | 2020-09-22 | 2026-04-12 | Frozen (legacy)                  |
| AAVEUSDT |  8,286,114 | 2020-10-15 | 2026-04-12 | Frozen (legacy)                  |
| UNIUSDT  |  8,233,056 | 2020-09-17 | 2026-04-11 | Frozen (legacy)                  |
| LINKUSDT |  7,911,352 | 2019-01-16 | 2026-04-13 | Frozen (legacy)                  |
| FILUSDT  |  7,369,072 | 2020-10-15 | 2026-04-12 | Frozen (legacy)                  |
| BCHUSDT  |  5,711,842 | 2019-11-28 | 2026-04-13 | Frozen (legacy)                  |
| SUIUSDT  |  3,631,844 | 2023-05-03 | 2026-04-11 | Frozen (legacy)                  |
| LTCUSDT  |  1,861,561 | 2018-01-16 | 2026-04-12 | Frozen (legacy)                  |
| NEARUSDT |    165,282 | 2020-10-14 | 2026-04-12 | Frozen (legacy)                  |

---

## Layer 2 — Inactive CH parts (dropped today, GC-pending)

```
SELECT count() AS parts, formatReadableSize(sum(bytes_on_disk)) AS size,
       toDate(modification_time) AS dropped_date
FROM system.parts
WHERE database='opendeviationbar_cache' AND active=0
GROUP BY dropped_date

→ 4,152 parts / 55.10 GiB / dropped 2026-05-01
```

Top dropped partitions by size (all from today's drop operation):

| Partition (symbol, threshold_dbps)                | Parts |     Size |
| ------------------------------------------------- | ----: | -------: |
| (DOGEUSDT, 100)                                   |   424 | 7.48 GiB |
| (XRPUSDT, 100)                                    |   320 | 4.35 GiB |
| (ADAUSDT, 100)                                    |   200 | 4.00 GiB |
| (SOLUSDT, 100)                                    |   280 | 3.78 GiB |
| (LINKUSDT, 100)                                   |   212 | 3.75 GiB |
| (BNBUSDT, 100)                                    |   132 | 3.63 GiB |
| (UNIUSDT, 100)                                    |   236 | 3.20 GiB |
| (AVAXUSDT, 100)                                   |    88 | 3.18 GiB |
| (AAVEUSDT, 100)                                   |   136 | 3.09 GiB |
| (FILUSDT, 100)                                    |   216 | 2.87 GiB |
| (BCHUSDT, 100)                                    |   100 | 2.13 GiB |
| (DOGEUSDT, 250)                                   |   192 | 1.70 GiB |
| (SUIUSDT, 100)                                    |    56 | 1.40 GiB |
| ... (more partitions at 250, 500, 750 thresholds) |       |          |

**Time-sensitivity**: ClickHouse's merge-cleaner will permanently free this
data on its next pass (typically minutes-to-hours, depending on merge
policy). If recovery is desired, **ALTER TABLE ... ATTACH PARTITION
'(<symbol>, <threshold>)'** can re-attach individual partitions while
they're still on disk.

The fact that 55 GiB was dropped today suggests an explicit cleanup
operation. Whether it's the previous-deploy old-version data or the
current scope-narrowing churn is unclear from the timestamps (CH
records modification_time, not creation_time).

---

## Layer 3 — Binance source archives (raw, multi-year)

```
/home/tca/.cache/gapless-crypto-clickhouse/zips/
```

Monthly aggTrade zips by symbol:

| Symbol    | Months | Estimated coverage                  |
| --------- | -----: | ----------------------------------- |
| BTCUSDT   |    118 | ~10 years (2021-01 through current) |
| ETHUSDT   |    118 | ~10 years                           |
| BNBUSDT   |    118 | ~10 years                           |
| ADAUSDT   |    118 | ~10 years                           |
| DOGEUSDT  |    118 | ~10 years                           |
| SOLUSDT   |    118 | ~10 years                           |
| XRPUSDT   |    118 | ~10 years                           |
| MATICUSDT |     79 | ~6.5 years                          |
| LINKUSDT  |     37 | ~3 years                            |
| AVAXUSDT  |     37 | ~3 years                            |

These are the **canonical inputs** that opendeviationbar-py processes into
ODB bars via `populate_cache_resumable()` or equivalent.

**Why unreachable from Flowsurface today**: not exposed; not directly
queryable via SQL. They flow through the bar-generation pipeline only
when the kintsugi-genesis gate is open.

---

## Layer 4 — opendeviationbar processing cache (113 GB)

```
/home/tca/.cache/opendeviationbar/
├── checkpoints/             # incremental processing state per (symbol, threshold)
├── repair_checkpoints/      # gap-fill operation state
├── ticks/                   # processed tick data (intermediate format)
├── telemetry/               # operational logs / heartbeat
├── checksum-registry.json   # data integrity ledger
└── ...
```

This is the kintsugi pipeline's working storage. With genesis enabled,
this cache + Layer 3 zips would regenerate Layers 1 and 2.

---

## The single gate

```bash
# deploy/opendeviationbar.env
OPENDEVIATIONBAR_KINTSUGI_GENESIS_ENABLED=false
```

Producer-side comment (from their session):

> "Disable kintsugi genesis backfill — only heal gaps BETWEEN existing bars.
> Prevents undoing clean-slate POC rebuilds.
> Set to true when ready for full history."

When flipped to `true`:

- kintsugi processes Layer 3 zips through the existing pipeline
- ODB bars get regenerated for all `enabled=true` symbols at all thresholds
- Layers 1+2 in CH expand back to the historical depth (multi-year)
- Flowsurface chart picks up the new bars on its next 1-second poll

When flipped back to `false`:

- New bar generation continues from live ingestion
- No re-ingestion of historical zips (preserving POC scope)

---

## Producer's stated rationale (from their session log)

> "BTCUSDT/ETHUSDT 33-day window came from `repair_direct_parquet.py
 --recent-days 30` (today's iter-11 deploy, v13.75.0). Older history
> was wiped intentionally during prior clean-slate POC rebuilds."

> "14 'frozen' symbols have `enabled = false` in symbols.toml. Only
> BTCUSDT + ETHUSDT have `enabled = true`."

> Decision on the 2026-05-01 handoff: **"Hold — no reply doc, no scope
> changes."**

---

## Summary

**The data exists.** Multiple layers, all on disk, all reproducible. The
chart shows what's active because that's what's been _deliberately_ exposed.

**The flip to expose multi-year history** is a single env-var change plus
an orchestration run. The producer team has the gate; they've chosen
to keep it closed for POC velocity.

**Time-pressure reminder**: the 55 GiB of inactive parts in Layer 2 will
be CH-merge-GC'd within a window probably measured in hours. If the team
later decides they want it back without re-running the full kintsugi
pipeline, that needs to happen sooner rather than later.

---

_Written 2026-05-01 by the flowsurface autoloop session in response to
"please look deep underneath the ClickHouse database and look at what's
really available and why we cannot get them." This document inventories
the four layers; the producer team owns the gate decision._

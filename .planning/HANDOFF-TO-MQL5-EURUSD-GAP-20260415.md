# Handoff to mql5 — EURUSD 9h gap 2026-04-14 23:57 → 2026-04-15 09:36 UTC

**From:** `~/fork-tools/flowsurface/` session (consumer)
**To:** `~/eon/mql5/` session (producer of fxview_cache.forex_bars)
**Date:** 2026-04-15
**Priority:** Diagnostic handoff — no blocker on flowsurface side, but asking for decision on repair

---

## What the user observed

Two adjacent ODB bars on EURUSD BPR0.5 chart:

```
Bar N  (last before gap):
  OPEN 1.17980  HIGH 1.17988  LOW 1.17979  CLOSE 1.17988
  2026-04-14 23:56:14.878 → 2026-04-14 23:57:14.444 UTC  (59.566s)

Bar N+1 (first after gap):
  OPEN 1.17817  HIGH 1.17818  LOW 1.17812  CLOSE 1.17812
  2026-04-15 09:35:40.346 → 2026-04-15 09:36:19.018 UTC  (38.672s)

Visible discontinuity: 9h 39min time gap + -14.5 dbps price discontinuity
```

On an index-based ODB chart these render as visually-adjacent bars with a vertical price cliff between them, which looks like a data bug to a user inspecting history.

---

## Forensic findings across all three layers

### Layer 1 — ClickHouse (consumer, faithful mirror)

At `threshold=5`, there are **zero EURUSD bars** between 23:57:14 Apr 14 and 09:35:40 Apr 15 UTC. At `threshold=10`, exactly **1 bar** exists somewhere in that window. CH is not lossy — it's faithfully storing what the sidecar wrote.

Interesting: the adjacent-bar `sidecar_version` field tells the story — last-before-gap was written by `1.23.2+fb83bb2-dirty`, first-after-gap was written by `1.22.0+7b96499-dirty` (a DIFFERENT version, not a later one). Meaning: the bar stream crossed a sidecar version discontinuity without bars being written in between.

### Layer 2 — Parquet source files (partial damage)

File inventory for Apr 14-15 in `/home/tca/.mt5/drive_c/Program Files/MetaTrader 5/tick_data/EURUSD/2026/`:

| File                                |   Size |    Rows | Notes                                            |
| ----------------------------------- | -----: | ------: | ------------------------------------------------ |
| `EURUSD_20260414.parquet`           |  13 KB |   2,613 | Clean, 246 ticks overlap gap start (23:57–23:59) |
| `EURUSD_20260414_1.parquet`         | 606 KB | 156,388 | Clean, full-day, closed 00:09 Apr 15             |
| `EURUSD_20260414_2.parquet`         |   2 KB |       1 | Closeout stub at 08:16 Apr 15                    |
| `EURUSD_20260414_3.parquet`         |   2 KB |       1 | Closeout stub at 08:42 Apr 15                    |
| `EURUSD_20260415.parquet`           | 851 KB |       — | **CORRUPT** — "magic bytes not found in footer"  |
| `EURUSD_20260415_partial.parquet`   |    0 B |       — | Empty                                            |
| `EURUSD_20260415_partial_1.parquet` |    0 B |       — | Empty                                            |
| `EURUSD_20260415_partial_2.parquet` | 237 KB |       — | **CORRUPT** — parseable magic bytes missing      |

The `_partial_*` naming pattern + 0-byte/corrupt files are the fingerprint of MT5's tick-writer DLL being **interrupted mid-flush** (process killed before the Parquet footer was finalized). The Apr 14 main file closed cleanly at 00:09 (post-midnight rollover worked), but the Apr 15 files show cycling without proper finalization.

### Layer 3 — MT5 + systemd event timeline

```
Apr 14 23:58:51  systemd: fxview-sidecar STOPPED
Apr 14 23:58:52  sidecar RESTARTED as 1.15.0+ce3c88c-dirty  (downgrade from 1.23.2)
Apr 14 23:59:51  sidecar RESTARTED as 1.15.0+92e2019  (same minor, different SHA)
Apr 15 00:04    opendeviationbar session-enricher still processing EURUSD (quiet but alive)
Apr 15 00:07:44 mt5.service CRASHED — exit code 1, FAILURE
Apr 15 00:08:00 mt5.service restarted
Apr 15 00:09:00 [tick-writer] WAL replay: recovered 1143 ticks from EURUSD_20260415.arrows
Apr 15 03:26:45 sidecar RESTARTED as 1.18.2+af522f6-dirty  (yet another version)
Apr 15 03:46:16 sidecar STOPPED
Apr 15 03:53:17 sidecar TRIED to start as 1.19.0+8dfd616-dirty
                → FAILED: "ring file I/O error: No such file or directory"
Apr 15 03:53:17 systemd restart counter = 1
...repeated 5+ times through 03:53:43+...
Apr 15 ~09:35   recovery — first bar after gap written by 1.22.0+7b96499-dirty
```

**Five sidecar version changes in ~10 hours**: `1.23.2` → `1.15.0+ce3c88c` → `1.15.0+92e2019` → `1.18.2` → `1.19.0` (crash loop) → `1.22.0`. No version increment is monotonic; the stream crossed through downgrades and sideways jumps.

---

## Root cause

**Concurrent deploy-storm on bigblack combined with one MT5 process crash.** Not a data-model bug, not an architectural issue, not broker-side. It's the signature of rapid iteration where:

1. Each sidecar restart orphaned unconsumed ticks from the memory-mapped ring
2. The `_1.19.0+8dfd616-dirty` version crash-looped because whoever deleted/moved the ring file didn't recreate it before the restart
3. MT5's tick-writer DLL kept cycling new `_partial_N.parquet` files but the enclosing MT5 process was being force-restarted before the DLL could write valid Parquet footers

---

## What we ask you to decide

The flowsurface user's question: **"Should we do a full clean slate rebuild?"**

Our honest assessment: **no, we don't think so** — but we defer to you. Reasoning:

### Why we're skeptical of a 3rd clean slate in 24 hours

- You already clean-slated on 2026-04-14 evening (per `.planning/HANDOFF-FXVIEW-FOREX-PIPELINE.md`)
- You re-backfilled after the Portcullis fix (`fb83bb2`)
- You re-backfilled after the BarBuilder gap-reset fix
- Each wipe costs all accumulated live-streamed data (hours of forward-only ticks that can't be reproduced from Parquet because the ring was the delivery mechanism)
- A third clean slate doesn't address the root cause (version churn during live streaming)

### Surgical option (our recommendation)

1. **Re-emit only the gap window** from whatever Parquet is recoverable:
   - `EURUSD_20260414.parquet` + `EURUSD_20260414_1.parquet` cover 23:57–23:59 Apr 14 UTC (~3 min, 246 ticks)
   - Apr 15 00:00–09:35 is **unrecoverable** — all 4 Parquet files are either 0-byte or corrupt at the footer. No backfill can reconstruct what was never persisted.
2. **Accept the ~9h lacuna** as an operational scar and move on. Flowsurface can surface it visually (dashed line + "9h 39min data gap" label) so users don't mistake it for flat-market behavior.

### What we're asking for

1. **Confirm the above is a fair characterization** of what happened. If our timeline is wrong, tell us.
2. **Decide**: full clean slate, surgical re-emit, or accept the gap.
3. **Versioning discipline going forward** — this is the highest-leverage ask:
   - Don't `systemctl restart fxview-sidecar` while live-streaming unless the new build has been validated against a staging ring
   - Avoid downgrades on the live-streaming path; if a rollback is needed, stop the stream, rebuild parquet coverage, then resume
   - Consider capturing `sidecar_version` transitions in a small audit table (1 row per restart event with old_ver, new_ver, timestamp) so future gap forensics can be done in <1 query

---

## What flowsurface will do regardless

- Nothing destructive. The chart correctly renders whatever is in `fxview_cache.forex_bars`.
- **Proposed small addition** (not gated on this handoff): render a dashed vertical marker between any two consecutive ODB bars where `open_time_ms[N+1] - close_time_ms[N] > 10 min`, labeled with the gap duration. This makes operational gaps legible so users don't misread them as data bugs. ~30 LOC, already scoped.

---

## Appendix — commands to reproduce the investigation on bigblack

```bash
# Layer 1: CH bar adjacency
clickhouse-client -q "
WITH edge AS (
  (SELECT close_time_us, open, close, quote_count, sidecar_version
   FROM fxview_cache.forex_bars
   WHERE symbol='EURUSD' AND threshold_decimal_bps=5
     AND close_time_us < toInt64(toUnixTimestamp(toDateTime('2026-04-14 23:58:00','UTC'))*1e6)
   ORDER BY close_time_us DESC LIMIT 3)
  UNION ALL
  (SELECT close_time_us, open, close, quote_count, sidecar_version
   FROM fxview_cache.forex_bars
   WHERE symbol='EURUSD' AND threshold_decimal_bps=5
     AND close_time_us > toInt64(toUnixTimestamp(toDateTime('2026-04-15 09:35:00','UTC'))*1e6)
   ORDER BY close_time_us ASC LIMIT 3)
)
SELECT toDateTime64(fromUnixTimestamp64Micro(close_time_us), 3, 'UTC'),
       round(open, 5), round(close, 5), quote_count, sidecar_version
FROM edge ORDER BY close_time_us FORMAT PrettyCompactMonoBlock
"

# Layer 2: Parquet integrity
ls -la "/home/tca/.mt5/drive_c/Program Files/MetaTrader 5/tick_data/EURUSD/2026/" | grep 2026041[45]

# Layer 3: systemd journal
journalctl --user -u fxview-sidecar.service --since "2026-04-14 23:55 UTC" --until "2026-04-15 09:40 UTC" --no-pager
journalctl --user -u mt5.service --since "2026-04-15 00:00 UTC" --until "2026-04-15 01:00 UTC" --no-pager

# MT5 EA log (UTF-16, needs iconv)
iconv -f UTF-16 -t UTF-8 "/home/tca/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Logs/20260415.log" | grep EURUSD
```

---

_Written 2026-04-15 by the flowsurface session after user-led forensic investigation triggered by a visually-cliff gap between adjacent EURUSD ODB bars._

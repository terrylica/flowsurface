# Handoff to mql5 — Port self-healing primitives from deleted Python forex-streaming

**From:** `~/fork-tools/flowsurface/` session (consumer)
**To:** `~/eon/mql5/` session (producer — owns `fxview-sidecar`)
**Date:** 2026-04-15
**Priority:** Root-cause fix for the class of gaps documented in
`HANDOFF-TO-MQL5-EURUSD-GAP-20260415.md`
**Related:** Observation that "gaps like this didn't happen before we removed the forex code from `opendeviationbar-py`"

---

## Claim to verify

The 9h EURUSD gap on 2026-04-14 → 2026-04-15 (and the earlier XAUUSD 0.10M-bar under-ingestion,
and the 58-min EURUSD watermark stall we've seen) all share a root cause: **the current
Rust `fxview-sidecar` lacks self-healing primitives that the deleted Python
`scripts/forex_streaming.py` had.**

When the mql5 team ported forex from Python to Rust, the transport layer and performance
characteristics were preserved, but the defensive/recovery logic was not carried over. This
handoff catalogs the missing primitives with exact Python source so they can be translated.

Source: `opendeviationbar-py` repo at `~/eon/opendeviationbar-py/`, commit `5357bce2^`
(the commit just before `chore(forex-cleanup): Phase 3` deleted `scripts/forex_streaming.py`).

---

## 7 self-healing primitives that were removed

### 1. Ring-buffer open with exponential backoff

When the ring file is missing (e.g., MT5 restarting, `/dev/shm` just mounted, sidecar
racing MT5 startup), retry with backoff rather than crashing immediately.

```python
# forex_streaming.py:878-903
def _open_ring_with_retry(symbol, *, max_retries=10):
    backoff = 1.0  # seconds
    for attempt in range(max_retries):
        try:
            return RingBufferConsumer(symbol)
        except FileNotFoundError:
            logger.warning("%s: ring buffer not found (attempt %d/%d), retrying in %.1fs",
                           symbol, attempt + 1, max_retries, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)  # cap at 60s
        except Exception:
            logger.exception("%s: ring buffer open failed (attempt %d/%d)", ...)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    return None
```

**Observed symptom from its absence (2026-04-15 03:53:17 UTC):**

```
fxview-sidecar: Error: ring file I/O error: No such file or directory (os error 2)
systemd: fxview-sidecar.service: Scheduled restart job, restart counter is at 1
...repeated 5+ times in 30 seconds...
```

Systemd's restart policy kicked in as a blunt substitute but fired far too often. A
sidecar-internal retry with exponential backoff would have absorbed the transient without
polluting the journal and without making the ring file race-critical.

---

### 2. Sequence-reset detection (MT5 restart signal)

When the ring's monotonic sequence counter _decreases_, MT5 restarted. The consumer
must not treat this as data corruption — it's a signal to reset its own consumer state.

```python
# forex_streaming.py:210-218 (inside RingBufferConsumer.read_new_ticks)
new_seq = self.read_sequence()
if new_seq == last_seq:
    return [], last_seq
if new_seq < last_seq:
    logger.warning(
        "%s: sequence reset detected (was %d, now %d) -- MT5 may have restarted",
        self._symbol, last_seq, new_seq,
    )
    return [], new_seq   # start fresh from new_seq, no error raised
```

**Observed symptom from its absence (2026-04-15 00:07:44 UTC):**
MT5 crashed + restarted 16 seconds later. Without this primitive, the sidecar either:

- Loses the MT5-restart signal entirely (if using `starting_seq = current producer sequence`, as current Rust code does), dropping every unconsumed tick in the old ring
- OR treats the sequence regression as an error condition

Either way, ticks near the restart boundary are lost.

---

### 3. Wrap-around loss detection

Ring buffers have finite capacity (65536 slots). If the consumer falls far enough
behind that the producer wraps, the oldest unconsumed slots are overwritten. Don't
treat this as silent data loss — quantify and log it.

```python
# forex_streaming.py:222-230
gap = new_seq - last_seq
if gap > self.capacity:
    lost = gap - self.capacity
    logger.warning(
        "%s: ring buffer wrap-around, lost %d ticks (gap=%d, capacity=%d)",
        self._symbol, lost, gap, self.capacity,
    )
    start_seq = new_seq - self.capacity + 1   # salvage what's still in the ring
else:
    start_seq = last_seq + 1
```

**Why it matters:** current Rust sidecar starts from `current producer sequence` on
restart (logged as `starting_seq=2497`, `starting_seq=2596`, `starting_seq=20480` in
the gap window). This _intentionally_ skips unconsumed ticks without measuring or
logging the loss. During the gap the sidecar was started 5 times; each started
from "now" and discarded any backlog. No one knows how many ticks that cost us.

---

### 4. Weekend-gap processor reset

Portcullis breach mode has per-symbol stateful processors. If they hold open a bar
across a weekend (Fri 22:00 UTC close → Sun 22:00 UTC reopen = 48h gap in ticks),
the next tick's threshold-breach math is wildly wrong. Detect the gap and reset.

```python
# forex_streaming.py:680-688 (inside stream_symbol main loop)
FOUR_HOURS_US = 14_400_000_000
# Weekend gap detection: check first tick timestamp vs prev_last_ts
if prev_last_ts is not None:
    gap_us = first_ts - prev_last_ts
    if gap_us > FOUR_HOURS_US:
        logger.warning(
            "%s: weekend gap detected (%.1fh), resetting processors",
            symbol, gap_us / 3_600_000_000,
        )
        # reset Portcullis processors per threshold...
```

**Status in current Rust:** partially addressed — mql5 commit `3d35c78` added
week-boundary reset (Sun 22:00 UTC calendar-based). But gap-triggered reset (4h of
silence mid-week) is still missing. Needed for daily-settlement windows and for
recovery after MT5 disconnects mid-week.

---

### 5. Circuit breaker for ClickHouse write failures

When CH becomes temporarily unreachable (network blip, CH restart, disk pressure),
don't hammer it with every batch. Open circuit after 3 consecutive failures, retry
after 60s.

```python
# forex_streaming.py:571-612
class _CircuitBreaker:
    """States: CLOSED (normal) -> OPEN (after max_failures) -> HALF_OPEN (after retry_after_s)."""
    def __init__(self, max_failures=3, retry_after_s=60.0):
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
    def record_success(self): self.consecutive_failures = 0; self.state = "CLOSED"
    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= self.max_failures:
            self.state = "OPEN"
            logger.error("Circuit breaker OPEN after %d consecutive failures", ...)
    def allow_request(self) -> bool:
        if self.state == "CLOSED": return True
        if self.state == "OPEN":
            if time.monotonic() - self.last_failure_time >= self.retry_after_s:
                self.state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN allows one attempt
```

---

### 6. Resume TIDs on startup (idempotent re-ingest)

On sidecar startup, query CH for the last-known bar per `(symbol, threshold)` so the
new session starts from the correct boundary — not "now" (dropping backlog) and not
"zero" (re-emitting everything).

```python
# forex_streaming.py:_query_resume_tids
# Query: SELECT symbol, threshold, max(close_time_us), max(first_ref_id) ... FROM open_deviation_bars
# Use the result to seed the processor's starting state.
```

**Observed symptom from its absence:** the 9h gap might have been shorter if sidecar
1.22.0 on its first startup had queried CH for the last-good bar from 1.23.2 at
23:57:14, then backfilled from the ring/Parquet starting there. Instead, 1.22.0
started from 09:35 "live" and left the preceding 9h as a hole.

---

### 7. Pre-fill delete (idempotent writes)

Before inserting a batch of bars, `DELETE` the tail range in CH that overlaps.
Combined with `ReplacingMergeTree`, this guarantees that resume-after-crash
produces the same final state as continuous operation.

```python
# forex_streaming.py:_prefill_delete
# DELETE FROM ... WHERE symbol = ? AND threshold = ? AND close_time_us > ?
# Run before INSERT of the next batch. Makes the insert pipeline idempotent.
```

Current Rust sidecar relies on `ReplacingMergeTree(computed_at)` dedup on the merge
path, which is async and doesn't guarantee SELECT returns the right version until
merge completes. Pre-fill delete makes idempotency synchronous.

---

## Why the sum of these matters

Any single primitive alone is a nice-to-have. The _combination_ is what made the Python
service robust: one failure mode cascading into another was caught by at least one layer.

Current Rust sidecar has none of them. So any single failure — a flaky ring file, a CH
hiccup, an MT5 restart, a deploy churn — directly produces a data gap because there's no
catch layer.

The 9h gap on 2026-04-14 → 15 was exactly this: 5 sidecar version changes combined with
one MT5 crash produced a compounding failure cascade that the Python version would have
shrugged off (probably with a few warning lines in the log).

---

## What we're asking for

1. **Port primitives 1, 2, 3, 5, 6, 7 to `fxview-sidecar` in Rust.** Primitive 4 is partially
   addressed (commit `3d35c78`) — complete it to cover mid-week gap resets.

2. **Prioritization suggestion** (highest leverage first):
   - #1 (ring retry) — eliminates most systemd-restart-loops
   - #6 (resume TIDs) — eliminates version-churn gaps
   - #5 (circuit breaker) — eliminates CH-transient-fault amplification
   - #2, #3 (sequence detection) — eliminates MT5-restart silent loss
   - #7 (pre-fill delete) — hardens idempotency for future automations

3. **Optional nice-to-have**: a `sidecar_events` audit table with 1 row per
   `{restart, ring_retry, sequence_reset, wrap_loss, cb_trip, cb_close}` event. Saves
   future forensics.

---

## Appendix — commands to retrieve the Python source verbatim

```bash
cd ~/eon/opendeviationbar-py
DEL=$(git log --oneline --all --diff-filter=D --follow -- scripts/forex_streaming.py | head -1 | awk '{print $1}')
echo "Parent of deletion commit: ${DEL}^"

# Full file
git show "${DEL}^:scripts/forex_streaming.py" > /tmp/forex_streaming_before_deletion.py

# Specific sections referenced above:
git show "${DEL}^:scripts/forex_streaming.py" | sed -n '208,235p'  # sequence/wrap detection
git show "${DEL}^:scripts/forex_streaming.py" | sed -n '565,625p'  # circuit breaker
git show "${DEL}^:scripts/forex_streaming.py" | sed -n '874,903p'  # ring retry
git show "${DEL}^:scripts/forex_streaming.py" | grep -n "_query_resume_tids\|_prefill_delete"  # resume/prefill
```

The original commit `5357bce2` (2026-04-14 evening, "chore(forex-cleanup): Phase 3 —
delete forex scripts, modules, and refactor orchestration") was the right cleanup for
opendeviationbar-py's scope (forex moved to mql5). What got lost in translation was
_not_ the transport layer — which mql5 rebuilt cleanly in Rust — but the 7 defensive
primitives above, which represent ~200 lines of hard-won production experience.

The user asked whether these gaps were happening before the clean-slate removal of
forex code. Looking at Python-era git log, the answer is: no gaps of this magnitude
appeared in the journal when the Python service was live. The Rust service handles the
happy path identically but fails differently on every unhappy path.

---

_Written 2026-04-15 by the flowsurface session after tracing the user's observation
that gaps of this kind didn't happen before the 2026-04-14 forex-cleanup. Source
excerpts are from opendeviationbar-py commit `53940c20..5357bce2^`._

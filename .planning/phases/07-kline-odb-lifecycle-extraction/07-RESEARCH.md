# Phase 7: Kline ODB Lifecycle Extraction - Research

**Researched:** 2026-03-27
**Domain:** Rust module extraction -- ODB lifecycle orchestration from kline/mod.rs
**Confidence:** HIGH

## Summary

Phase 7 extracts ODB lifecycle orchestration code from `kline/mod.rs` into a new `kline/odb_lifecycle.rs` submodule. The `invalidate()` method currently contains ~343 LOC of ODB-specific lifecycle logic (watchdog, sentinel, viewport digest, telemetry snapshot) that is entirely separate from chart rendering and canvas interaction. This code accesses `&mut self` fields directly -- no RefCell borrows are involved in the extractable blocks, making this a mechanically safe extraction.

The critical finding is that the 4 extractable blocks in `invalidate()` (343 LOC) plus relocating relevant tests (~60-80 LOC) will achieve the 361+ LOC reduction needed. The extraction follows the exact same `use super::*` + `impl KlineChart` pattern established by `odb_core.rs` and `data_ops.rs`.

**Primary recommendation:** Extract the 4 ODB lifecycle blocks from `invalidate()` into helper methods in `odb_lifecycle.rs`, then call those helpers from `invalidate()`. Keep `invalidate()` in mod.rs as the orchestrator that calls lifecycle helpers.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- all implementation choices are at Claude's discretion (infrastructure phase).

### Claude's Discretion

All implementation choices are at Claude's discretion. CRITICAL: Preserve existing RefCell borrow discipline. Never hold an immutable borrow across a borrow_mut() call. Runtime panics, not compile errors.

### Deferred Ideas (OUT OF SCOPE)

None.
</user_constraints>

<phase_requirements>

## Phase Requirements

| ID       | Description                                                                                                                | Research Support                                                                                                                                                                                            |
| -------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KLINE-02 | ODB lifecycle orchestration (gap-fill, reconciliation, sidecar) extracted to `kline/odb_lifecycle.rs` (~600 LOC reduction) | Identified 4 extractable blocks totaling ~343 LOC from `invalidate()` plus ~60-80 LOC of related tests. The 600 LOC target in requirements was aspirational; actual extractable lifecycle code is ~400 LOC. |
| KLINE-03 | kline/mod.rs reduced below 1800 LOC after extractions                                                                      | Current: 2160 LOC. Extracting ~370+ LOC brings it to ~1790 or below 1800.                                                                                                                                   |
| VER-01   | `cargo clippy -- -D warnings` passes after phase                                                                           | Baseline verified clean. Mechanical extraction preserves all types.                                                                                                                                         |
| VER-02   | Zero behavior changes                                                                                                      | Exact code copy with method delegation -- no logic modifications.                                                                                                                                           |
| VER-03   | No new `unsafe` code introduced                                                                                            | No unsafe needed for module extraction.                                                                                                                                                                     |

</phase_requirements>

## Architecture Patterns

### Extraction Target Analysis

**File:** `src/chart/kline/mod.rs` (2160 LOC)

The `invalidate()` method (lines 919-1378, 460 LOC) contains 4 ODB-specific lifecycle blocks that are self-contained and access only `&mut self` fields:

| Block              | Lines     | LOC | Purpose                                                        | Fields Accessed                                                                                                                                |
| ------------------ | --------- | --- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Watchdog           | 1025-1058 | 34  | Trade feed dead-man's switch (90s timeout)                     | `last_trade_received_ms`, `trade_feed_dead_alerted`, `last_ws_agg_trade_id`, `ch_reconcile_count`                                              |
| Sentinel           | 1060-1240 | 181 | Bar continuity audit (60s) + Telegram alerts + refetch trigger | `chart.basis`, `fetching_trades`, `last_sentinel_audit`, `sentinel_gap_count`, `sentinel_refetch_pending`, `sentinel_healable_gap_min_time_ms` |
| Viewport digest    | 1242-1323 | 82  | Periodic bar quality summary (60s)                             | `chart.basis`, `last_viewport_digest`, `data_source`, `odb_processor`, `ch_reconcile_count`                                                    |
| Telemetry snapshot | 1325-1370 | 46  | ChartSnapshot emission (30s)                                   | `chart.basis`, `last_snapshot`, `data_source`, `odb_processor`, `odb_completed_count`                                                          |

**Total extractable from invalidate(): ~343 LOC**

**Tests section** (lines 1952-2160, 209 LOC): Contains tests for gap-fill guards, dedup fences, cooldown arithmetic, and gap detection. ~60-80 LOC of these tests are specifically testing ODB lifecycle guard logic and should move with the lifecycle code.

### Recommended Extraction Strategy

**Pattern: Helper methods called from invalidate()**

Do NOT move `invalidate()` itself. Instead, extract each lifecycle block into a `pub(super)` method on KlineChart in `odb_lifecycle.rs`, then replace the inline code in `invalidate()` with a method call.

```rust
// In odb_lifecycle.rs
impl KlineChart {
    /// Trade feed liveness watchdog (dead-man's switch, 90s threshold).
    pub(super) fn check_trade_feed_watchdog(&mut self) { ... }

    /// Sentinel: bar-level agg_trade_id continuity audit (every 60s, ODB only).
    pub(super) fn run_sentinel_audit(&mut self, now: Instant) { ... }

    /// Viewport digest: periodic bar quality summary (every 60s, ODB only).
    pub(super) fn emit_viewport_digest(&mut self, now: Instant) { ... }

    /// Telemetry snapshot: emit ChartSnapshot (every 30s, ODB only).
    #[cfg(feature = "telemetry")]
    pub(super) fn emit_telemetry_snapshot(&mut self, now: Instant) { ... }
}
```

```rust
// In mod.rs invalidate(), replace inline blocks with:
self.check_trade_feed_watchdog();

if self.chart.basis.is_odb() && let Some(t) = now {
    self.run_sentinel_audit(t);
    self.emit_viewport_digest(t);
}

#[cfg(feature = "telemetry")]
if let Some(t) = now {
    self.emit_telemetry_snapshot(t);
}
```

### Recommended Project Structure (post-extraction)

```
src/chart/kline/
├── mod.rs            # KlineChart struct, new(), invalidate() (orchestrator), canvas::Program impl (~1790 LOC)
├── bar_selection.rs  # Bar range selection state and rendering
├── crosshair.rs      # Crosshair tooltip rendering
├── data_ops.rs       # Data insertion, aggregation queries, missing_data_task
├── odb_core.rs       # ODB core: trade insertion, CH reconciliation, gap-fill, bar continuity audit
├── odb_lifecycle.rs  # NEW: ODB lifecycle: watchdog, sentinel orchestration, viewport digest, telemetry
└── rendering.rs      # Candle/footprint/cluster rendering
```

### Anti-Patterns to Avoid

- **Moving invalidate() itself**: `invalidate()` is the `Chart` trait impl and must stay in mod.rs. Extract helper methods, not the orchestrator.
- **Introducing new RefCell borrows**: The extractable blocks use `&mut self` -- keep it that way. No RefCell needed.
- **Splitting the sentinel block at arbitrary points**: The sentinel block (181 LOC) has tight internal coupling (anomaly classification, Telegram formatting, refetch triggering). Extract it as one method.
- **Moving autoscale/cache-clear logic**: Lines 919-1024 (autoscale + cache clearing) are chart-generic, not ODB lifecycle. Leave them in invalidate().

## Don't Hand-Roll

| Problem                   | Don't Build           | Use Instead                     | Why                                                  |
| ------------------------- | --------------------- | ------------------------------- | ---------------------------------------------------- |
| Module extraction pattern | Custom import scheme  | `use super::*`                  | Established pattern in odb_core.rs and data_ops.rs   |
| Conditional compilation   | Manual feature checks | `#[cfg(feature = "telemetry")]` | Telemetry snapshot block is already behind this gate |

## Common Pitfalls

### Pitfall 1: Breaking the invalidate() return chain

**What goes wrong:** `invalidate()` returns `Option<Action>` from `self.missing_data_task()` at its tail. Extracting blocks must not break this return path.
**Why it happens:** The extracted blocks don't return values -- they are fire-and-forget side effects. But inserting method calls must not shadow the tail return.
**How to avoid:** Place extracted method calls in the same positions as the original inline code. The tail `self.missing_data_task()` call remains unchanged.
**Warning signs:** `invalidate()` returning `None` when it should return an Action (data stops loading).

### Pitfall 2: Sentinel audit_bar_continuity() lives in odb_core.rs

**What goes wrong:** The sentinel block in invalidate() calls `self.audit_bar_continuity()` which is defined in odb_core.rs (line 700). The extracted `run_sentinel_audit()` in odb_lifecycle.rs must be able to call it.
**Why it happens:** Methods from different submodules are all on `impl KlineChart` so they can call each other via `self.method()`.
**How to avoid:** `use super::*` imports bring all types. `pub(super)` visibility on `audit_bar_continuity()` already allows cross-submodule calls. No changes needed.
**Warning signs:** Compilation error "method not found" -- indicates visibility issue.

### Pitfall 3: cfg(feature = "telemetry") conditionality

**What goes wrong:** The telemetry snapshot block (lines 1325-1370) is behind `#[cfg(feature = "telemetry")]`. The extracted method and its call site must both preserve this gate.
**Why it happens:** Feature-gated code compiles away when feature is disabled. Both the method definition AND the call must be gated.
**How to avoid:** Apply `#[cfg(feature = "telemetry")]` to both the method in odb_lifecycle.rs and the call site in invalidate().
**Warning signs:** Build failure when `telemetry` feature is disabled.

### Pitfall 4: Test relocation -- GapFillRequest import path

**What goes wrong:** Tests import `super::GapFillRequest` which is re-exported from odb_core. If tests move to odb_lifecycle.rs, the import path changes.
**Why it happens:** `super::` resolves differently depending on which module the test lives in.
**How to avoid:** Tests that test lifecycle guard logic move to odb_lifecycle.rs; tests that test gap-fill/dedup stay in mod.rs. Each test module imports from the correct relative path.
**Warning signs:** `use super::GapFillRequest` fails in the new location.

### Pitfall 5: last_sentinel_audit field is Instant (not millisecond timestamp)

**What goes wrong:** Sentinel uses `Instant` for timing, watchdog uses `u64` millisecond timestamps. Mixing them up causes subtle bugs.
**Why it happens:** Different subsystems chose different time representations.
**How to avoid:** Method signatures make it clear: sentinel/viewport/telemetry take `now: Instant`; watchdog reads `SystemTime::now()` internally.
**Warning signs:** Type mismatch errors at compile time (Rust catches this).

## RefCell Borrow Analysis (CRITICAL)

**Finding: No RefCell borrows in extractable code.** All RefCell borrows in mod.rs are confined to:

- `canvas::Program::update()` (lines 1393-1548) -- bar_selection interaction
- `canvas::Program::draw()` (lines 1836, 1850) -- bar_selection read
- `canvas::Program::mouse_interaction()` (line 1899) -- bar_selection read

The 4 lifecycle blocks in `invalidate()` access only `&mut self` fields directly. No RefCell discipline changes needed. Confidence: HIGH.

## Code Examples

### Extraction pattern (from odb_core.rs -- established convention)

```rust
// odb_lifecycle.rs
// ODB lifecycle orchestration: watchdog, sentinel, viewport digest, telemetry.
// Extracted from kline/mod.rs to isolate fork-specific ODB complexity.
use super::*;

impl KlineChart {
    /// Trade feed liveness watchdog (dead-man's switch).
    /// Fires every frame via invalidate(). Alerts after 90s of no WS trades.
    pub(super) fn check_trade_feed_watchdog(&mut self) {
        if self.last_trade_received_ms > 0 {
            // ... exact code from lines 1029-1057 ...
        }
    }
}
```

### Call site pattern (in mod.rs invalidate())

```rust
// Replace lines 1025-1058 with:
self.check_trade_feed_watchdog();

// Replace lines 1060-1240 with:
if self.chart.basis.is_odb()
    && !self.fetching_trades.0
    && let Some(t) = now
    && t.duration_since(self.last_sentinel_audit) >= std::time::Duration::from_secs(60)
{
    self.run_sentinel_audit(t);
}
```

**Design choice:** The outer guard conditions (basis check, timer check) can stay in invalidate() or move into the helper. Recommend moving ALL guard conditions into the helper for maximum LOC reduction:

```rust
// Preferred: all logic in helper, invalidate() just calls
if let Some(t) = now {
    self.run_sentinel_audit(t);
    self.emit_viewport_digest(t);
}
self.check_trade_feed_watchdog();
```

This maximizes LOC reduction in mod.rs.

## LOC Budget Analysis

| Current | Action                             | LOC Change   | Result |
| ------- | ---------------------------------- | ------------ | ------ |
| 2160    | Extract watchdog block             | -34 +1 call  | 2127   |
| 2127    | Extract sentinel block             | -181 +1 call | 1947   |
| 1947    | Extract viewport digest            | -82 +1 call  | 1866   |
| 1866    | Extract telemetry snapshot         | -46 +1 call  | 1821   |
| 1821    | Move guard conditions into helpers | ~-25         | 1796   |
| 1796    | Move ~60 LOC of lifecycle tests    | -60          | ~1736  |

**Result: ~1736-1796 LOC** -- comfortably under 1800 target.

Even without moving tests, moving guard conditions into helpers achieves ~1796 LOC (under 1800). The tests provide additional margin.

## Validation Architecture

### Test Framework

| Property           | Value                                                                 |
| ------------------ | --------------------------------------------------------------------- |
| Framework          | cargo test (Rust built-in)                                            |
| Config file        | `Cargo.toml` (workspace)                                              |
| Quick run command  | `cargo test -p flowsurface --lib -- chart::kline`                     |
| Full suite command | `cargo clippy --all-targets -- -D warnings && cargo test --workspace` |

### Phase Requirements to Test Map

| Req ID   | Behavior                                   | Test Type   | Automated Command                                   | File Exists?             |
| -------- | ------------------------------------------ | ----------- | --------------------------------------------------- | ------------------------ |
| KLINE-02 | ODB lifecycle methods callable from mod.rs | unit        | `cargo test -p flowsurface --lib -- chart::kline`   | Existing tests in mod.rs |
| KLINE-03 | mod.rs below 1800 LOC                      | smoke       | `wc -l src/chart/kline/mod.rs`                      | Manual check             |
| VER-01   | Clippy clean                               | lint        | `cargo clippy --all-targets -- -D warnings`         | N/A                      |
| VER-02   | No behavior change                         | integration | Full app launch + ODB pane loads                    | Manual                   |
| VER-03   | No unsafe                                  | grep        | `grep -r "unsafe" src/chart/kline/odb_lifecycle.rs` | N/A                      |

### Sampling Rate

- **Per task commit:** `cargo clippy --all-targets -- -D warnings && cargo test -p flowsurface --lib -- chart::kline`
- **Per wave merge:** `cargo clippy --all-targets -- -D warnings && cargo test --workspace`
- **Phase gate:** Full suite green + `wc -l` verification

### Wave 0 Gaps

None -- existing test infrastructure covers all phase requirements.

## Sources

### Primary (HIGH confidence)

- Direct code analysis of `src/chart/kline/mod.rs` (2160 LOC)
- Direct code analysis of `src/chart/kline/odb_core.rs` (1524 LOC) -- extraction pattern reference
- Direct code analysis of `src/chart/kline/data_ops.rs` (246 LOC) -- extraction pattern reference
- `cargo clippy` baseline verified clean

### Secondary (MEDIUM confidence)

- Phase 6 PLAN.md -- extraction pattern and approach verified successful

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - pure Rust module extraction, no new dependencies
- Architecture: HIGH - extraction pattern established by odb_core.rs and data_ops.rs in prior phases
- Pitfalls: HIGH - RefCell analysis is definitive (grep-verified), LOC budget is arithmetic

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable -- no external dependencies)

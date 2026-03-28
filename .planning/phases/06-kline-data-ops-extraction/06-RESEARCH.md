# Phase 6: Kline Data Ops Extraction - Research

**Researched:** 2026-03-27
**Domain:** Rust module extraction -- data insertion/query methods from kline/mod.rs
**Confidence:** HIGH

## Summary

Phase 6 extracts data insertion and aggregation query methods from `src/chart/kline/mod.rs` (currently 2390 LOC) into a new `kline/data_ops.rs` submodule. The target is ~200 LOC reduction (KLINE-01), contributing toward the Phase 7 goal of getting kline/mod.rs below 1800 LOC (KLINE-03).

The extraction candidates are pure data-operation methods that do not touch canvas rendering or `RefCell` interior mutability. Five methods totaling ~246 LOC are identified: `missing_data_task` (143 LOC), `calc_qty_scales` (38 LOC), `toggle_indicator` (29 LOC), `insert_hist_klines` (22 LOC), and `insert_open_interest` (14 LOC). All are inherent `impl KlineChart` methods that take `&mut self` and interact with `data_source`, `indicators`, and `request_handler` -- they can be extracted as methods in a separate `impl KlineChart` block in `data_ops.rs` using `pub(super)` visibility for fields accessed from the new module.

The existing codebase already has `odb_core.rs` as a private submodule pattern for KlineChart -- the same approach works for `data_ops.rs`. ODB-specific data insertion (`insert_trades`, `insert_raw_trades`, `insert_odb_hist_klines`, `update_latest_kline`) is already in `odb_core.rs` and out of scope.

**Primary recommendation:** Extract the 5 identified methods as an `impl KlineChart` block in `kline/data_ops.rs`, using the same submodule pattern as `odb_core.rs`. No field visibility changes needed -- Rust allows `impl` blocks in submodules to access private fields of the parent module's struct.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- all implementation choices at Claude's discretion (infrastructure phase).

### Claude's Discretion

All implementation choices are at Claude's discretion -- pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions. Note: kline/mod.rs already uses RefCell<T> for interior mutability -- preserve existing borrow discipline.

### Deferred Ideas (OUT OF SCOPE)

None -- infrastructure phase.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID       | Description                                                                                    | Research Support                                                                                                                                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| KLINE-01 | Data operation methods (TickAggr access, aggregation queries) extracted to `kline/data_ops.rs` | 5 methods identified: `missing_data_task` (143), `calc_qty_scales` (38), `toggle_indicator` (29), `insert_hist_klines` (22), `insert_open_interest` (14) = 246 LOC. All are pure data ops with no canvas/rendering dependency. |
| VER-01   | `cargo clippy -- -D warnings` passes after every phase                                         | Standard verification gate -- run `mise run lint` after extraction                                                                                                                                                             |
| VER-02   | Zero behavior changes -- all existing functionality works identically                          | Methods move file but stay on `KlineChart` -- all call sites unchanged. Historical kline loading, live trade insertion, OI insertion, indicator toggling, and missing-data fetch all route through same public API.            |
| VER-03   | No new `unsafe` code introduced                                                                | No unsafe needed for this extraction                                                                                                                                                                                           |

</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **No regressions**: `cargo clippy -- -D warnings` must pass clean
- **No behavior change**: All existing functionality must work identically
- **Incremental**: Each phase independently shippable
- **Rust edition**: 2024, toolchain 1.93.1
- **Max line width**: 100 characters (rustfmt.toml)
- **Lint**: `mise run lint` = `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings`
- **FILE-SIZE-OK comments**: Line 1 of kline/mod.rs has `// FILE-SIZE-OK: upstream file, splitting out of scope for this fork` -- update after extraction to reflect reduced LOC
- **Fork patterns**: `// NOTE(fork):` prefix preserved; `// GitHub Issue:` links preserved
- **Visibility**: `pub(super)` for cross-submodule access within kline/ directory (same pattern as `odb_core.rs`)

## Architecture Patterns

### Current kline/mod.rs Structure (2390 LOC)

```
Lines 1-61:      Imports + mod declarations (bar_selection, crosshair, odb_core, rendering)
Lines 62-68:     BufferedChKline type alias
Lines 70-213:    Chart + PlotConstants trait impls
Lines 214-241:   make_indicator_with_config() free function
Lines 243-319:   KlineChart struct definition (77 fields!)
Lines 321-649:   KlineChart::new() (three Basis arms: Time/Tick/Odb)
Lines 651-654:   kind() accessor
Lines 655-797:   missing_data_task()              <<<< EXTRACT (143 LOC)
Lines 799-817:   raw_trades, set_handle, set_fetching_trades, clear_fetching_trades, tick_size
Lines 819-851:   study_configurator, update_study_configurator
Lines 853-889:   chart_layout, set_autoscale, set_include_forming, set_cluster_kind/scaling, basis
Lines 891-969:   change_tick_size, set_basis
Lines 971-1057:  studies/set_studies, config setters (ofi_ema, intensity, thermal, anomaly, sessions, keyboard_nav)
Lines 1058-1079: insert_hist_klines()             <<<< EXTRACT (22 LOC)
Lines 1081-1094: insert_open_interest()            <<<< EXTRACT (14 LOC)
Lines 1096-1133: calc_qty_scales()                 <<<< EXTRACT (38 LOC)
Lines 1135-1588: last_update, invalidate() (454 LOC -- NOT extracting, mixed rendering+monitoring)
Lines 1590-1618: toggle_indicator()                <<<< EXTRACT (29 LOC)
Lines 1621-2177: canvas::Program impl (update + draw + mouse_interaction -- rendering, NOT extracting)
Lines 2182-2390: #[cfg(test)] mod tests (209 LOC -- stays in mod.rs)
```

### Extraction Target: data_ops.rs (~246 LOC)

Methods to extract (all `impl KlineChart`):

| Method                 | Lines     | LOC | Visibility  | Dependencies                                                                                                           |
| ---------------------- | --------- | --- | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `missing_data_task`    | 655-797   | 143 | `fn` (priv) | `self.data_source`, `self.chart`, `self.request_handler`, `self.fetching_trades`, `self.indicators`, `self.sentinel_*` |
| `calc_qty_scales`      | 1096-1133 | 38  | `fn` (priv) | `self.data_source` (read-only)                                                                                         |
| `toggle_indicator`     | 1590-1618 | 29  | `pub fn`    | `self.indicators`, `self.kline_config`, `self.data_source`, `self.chart.layout`                                        |
| `insert_hist_klines`   | 1058-1079 | 22  | `pub fn`    | `self.data_source`, `self.raw_trades`, `self.indicators`, `self.request_handler`                                       |
| `insert_open_interest` | 1081-1094 | 14  | `pub fn`    | `self.request_handler`, `self.indicators`                                                                              |
| **Total**              |           | 246 |             |                                                                                                                        |

### Why These Methods and Not Others

**Included** -- Pure data operations:

- `insert_hist_klines`: Inserts historical klines into TimeSeries, updates indicators. No rendering.
- `insert_open_interest`: Routes OI data to indicator. No rendering.
- `missing_data_task`: Determines what data needs fetching. Returns `Option<Action>` for the caller to dispatch. No rendering.
- `calc_qty_scales`: Computes cluster quantity scales from data_source. Pure computation.
- `toggle_indicator`: Creates/destroys indicators, recalculates panel splits. Data management, not rendering.

**Excluded** -- Mixed concerns (rendering + data + monitoring):

- `invalidate` (454 LOC): Autoscale, cache clearing, watchdog, sentinel, viewport digest, telemetry. Too tangled for this phase; Phase 7 (KLINE-02) addresses this.
- `new` (328 LOC): Constructor. Not a data op.
- Config setters (`set_ofi_ema_period`, etc.): Small (5-15 LOC each), tightly coupled to config. Moving them gains little.
- `canvas::Program::update/draw` (556 LOC): Pure rendering/interaction.

### Submodule Pattern (from odb_core.rs precedent)

```rust
// In kline/mod.rs:
mod data_ops;   // No `pub use` needed -- methods are on KlineChart

// In kline/data_ops.rs:
use super::*;   // Brings all parent imports into scope

impl KlineChart {
    // Methods here can access all KlineChart fields (even private)
    // because they're in a submodule of the defining module.
    pub fn insert_hist_klines(&mut self, ...) { ... }
    pub fn insert_open_interest(&mut self, ...) { ... }
    pub fn toggle_indicator(&mut self, ...) { ... }
    fn missing_data_task(&mut self) -> Option<Action> { ... }
    fn calc_qty_scales(&self, ...) -> f32 { ... }
}
```

This is the exact same pattern used by `odb_core.rs` (lines 40-41 of mod.rs): `mod odb_core;` with methods implemented as `impl KlineChart` inside the submodule.

### Field Access Analysis

All 5 methods access `KlineChart` fields through `&self` or `&mut self`. Since `data_ops.rs` is a submodule of the module that defines `KlineChart`, it inherits visibility to all private fields -- no `pub(super)` changes needed on the struct fields.

Fields accessed by extracted methods:

| Field                               | Methods Using It                                            |
| ----------------------------------- | ----------------------------------------------------------- |
| `data_source`                       | All 5                                                       |
| `request_handler`                   | missing_data_task, insert_hist_klines, insert_open_interest |
| `indicators`                        | All except calc_qty_scales                                  |
| `chart` (ViewState)                 | missing_data_task, calc_qty_scales, toggle_indicator        |
| `fetching_trades`                   | missing_data_task                                           |
| `raw_trades`                        | insert_hist_klines                                          |
| `kline_config`                      | toggle_indicator                                            |
| `sentinel_refetch_pending`          | missing_data_task                                           |
| `sentinel_healable_gap_min_time_ms` | missing_data_task                                           |

### Import Dependencies for data_ops.rs

The extracted methods need these types (all available via `use super::*`):

- `Action`, `Basis`, `PlotData`, `request_fetch` -- from parent's `use super::{...}`
- `KlineChart` -- defined in parent
- `KlineIndicator`, `KlineDataPoint` -- from data crate
- `FetchRange`, `RequestHandler`, `is_trade_fetch_enabled` -- from connector
- `Kline`, `OpenInterest as OIData`, `Trade` -- from exchange crate
- `make_indicator_with_config` -- free function in parent mod.rs
- `indicator::kline::FetchCtx` -- from indicator module

A single `use super::*;` covers all of these since they're imported or defined in mod.rs.

### Caller Impact

All callers go through public methods on `KlineChart`. Moving methods to a submodule does not change the public API -- callers still write `chart.insert_hist_klines(...)`. No call site changes.

Internal callers within mod.rs:

- `invalidate()` calls `self.missing_data_task()` at line 1584 -- still works (method is on `self`)
- `draw()` calls `self.calc_qty_scales(...)` at line 1821 -- still works

## Don't Hand-Roll

| Problem              | Don't Build                 | Use Instead                              | Why                                                  |
| -------------------- | --------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| Module extraction    | Manual file copy-paste      | `use super::*` + impl block in submodule | Rust's module system handles visibility correctly    |
| Field access control | `pub(super)` on every field | Submodule inherits parent field access   | Submodules of the defining module see private fields |

## Common Pitfalls

### Pitfall 1: Breaking use super::\* with missing imports

**What goes wrong:** `data_ops.rs` uses `use super::*` but a type used by the extracted methods isn't imported in mod.rs's top-level scope -- it's imported inside the method body.
**Why it happens:** Some methods import types locally (e.g., `chrono::Utc` in `missing_data_task`).
**How to avoid:** Check each extracted method for local `use` statements or fully-qualified paths. Ensure they're either moved to `data_ops.rs` or kept as local imports within the method.
**Warning signs:** `cannot find type X in this scope` compiler errors.

### Pitfall 2: Circular call between mod.rs and data_ops.rs

**What goes wrong:** `invalidate()` in mod.rs calls `self.missing_data_task()` which is now in data_ops.rs. If data_ops.rs methods also call methods still in mod.rs, it creates a dependency cycle.
**Why it happens:** Rust allows this for `impl` blocks on the same type across submodules -- but it can be confusing.
**How to avoid:** This is actually fine in Rust. `impl` blocks on the same type can be in different modules. The compiler resolves method calls on `self` regardless of which file the impl block is in. No circular dependency issue.
**Warning signs:** None -- this is a non-issue in Rust.

### Pitfall 3: Forgetting to move the make_indicator_with_config dependency

**What goes wrong:** `toggle_indicator` calls `make_indicator_with_config()`, a free function in mod.rs. If it's not accessible from data_ops.rs, compilation fails.
**Why it happens:** Free functions in the parent module are accessible via `super::make_indicator_with_config` or through `use super::*`.
**How to avoid:** Verify that `make_indicator_with_config` is either `pub(super)` or imported via `use super::*`. Currently it has no visibility modifier (private to the module), but submodules inherit access to parent-module private items via `super::`.
**Warning signs:** `function make_indicator_with_config is private` -- won't happen since submodules can see parent privates.

### Pitfall 4: LOC counting error for Phase 7 target

**What goes wrong:** After extracting 246 LOC, mod.rs is 2390 - 246 = 2144 LOC. Phase 7 needs to get it below 1800, requiring 344+ more LOC from KLINE-02 (ODB lifecycle extraction, ~600 LOC target).
**Why it happens:** Accurate LOC counting matters for phase planning.
**How to avoid:** After extraction, verify actual LOC with `wc -l`. The 246 count includes blank lines and comments within method bodies.

## Code Examples

### data_ops.rs module structure

```rust
// src/chart/kline/data_ops.rs
//
// Data insertion and aggregation query methods extracted from kline/mod.rs.
// These methods operate on KlineChart's data_source, indicators, and
// request_handler -- no canvas rendering or RefCell interaction.

use super::*;

impl KlineChart {
    pub fn insert_hist_klines(&mut self, req_id: uuid::Uuid, klines_raw: &[Kline]) {
        // ... exact code from mod.rs lines 1058-1079
    }

    pub fn insert_open_interest(
        &mut self,
        req_id: Option<uuid::Uuid>,
        oi_data: &[OIData],
    ) {
        // ... exact code from mod.rs lines 1081-1094
    }

    pub fn toggle_indicator(&mut self, indicator: KlineIndicator) {
        // ... exact code from mod.rs lines 1590-1618
    }

    pub(super) fn missing_data_task(&mut self) -> Option<Action> {
        // ... exact code from mod.rs lines 655-797
    }

    pub(super) fn calc_qty_scales(
        &self,
        earliest: u64,
        latest: u64,
        highest: Price,
        lowest: Price,
        step: PriceStep,
        cluster_kind: ClusterKind,
    ) -> f32 {
        // ... exact code from mod.rs lines 1096-1133
    }
}
```

### mod.rs changes

```rust
// Add after existing mod declarations (line ~40):
mod data_ops;

// Remove: the 5 method bodies from the impl KlineChart block
// Keep: all other methods, struct definition, trait impls, canvas::Program impl
```

## Validation Architecture

### Test Framework

| Property           | Value                                       |
| ------------------ | ------------------------------------------- |
| Framework          | cargo test (built-in)                       |
| Config file        | Cargo.toml (workspace)                      |
| Quick run command  | `cargo test -p flowsurface -- kline::tests` |
| Full suite command | `cargo test --workspace`                    |

### Phase Requirements -> Test Map

| Req ID   | Behavior                              | Test Type | Automated Command                              | File Exists?                             |
| -------- | ------------------------------------- | --------- | ---------------------------------------------- | ---------------------------------------- |
| KLINE-01 | Data ops methods exist in data_ops.rs | build     | `cargo clippy --all-targets -- -D warnings`    | N/A (compile check)                      |
| VER-01   | Clippy clean                          | lint      | `mise run lint`                                | N/A                                      |
| VER-02   | Existing behavior preserved           | existing  | `cargo test --workspace`                       | Existing tests in mod.rs lines 2182-2390 |
| VER-03   | No new unsafe code                    | grep      | `grep -r 'unsafe' src/chart/kline/data_ops.rs` | N/A                                      |

### Sampling Rate

- **Per task commit:** `mise run lint`
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

None -- existing test infrastructure covers all phase requirements. The tests in mod.rs `#[cfg(test)] mod tests` continue to work unchanged (they test gap detection and guard logic, which stay in odb_core.rs/mod.rs). No new tests needed since this is a pure structural move with no behavior change.

## LOC Impact Analysis

| File                          | Before | After | Delta |
| ----------------------------- | ------ | ----- | ----- |
| `src/chart/kline/mod.rs`      | 2390   | ~2148 | -242  |
| `src/chart/kline/data_ops.rs` | 0      | ~255  | +255  |

Note: data_ops.rs is slightly larger than the extracted LOC due to the `use super::*;` import line and module-level comment. The net mod.rs reduction is ~242 LOC (some method boundaries have blank lines that stay or go).

Post-extraction mod.rs: ~2148 LOC. Phase 7 target: <1800 LOC. Remaining gap: ~348 LOC, well within KLINE-02's ~600 LOC target.

## Sources

### Primary (HIGH confidence)

- Direct codebase analysis of `src/chart/kline/mod.rs` (2390 LOC, read in full)
- Direct codebase analysis of `src/chart/kline/odb_core.rs` (submodule pattern precedent)
- Phase 5 RESEARCH.md (established extraction methodology)
- REQUIREMENTS.md (KLINE-01 definition: ~200 LOC reduction)

### Secondary (MEDIUM confidence)

- Rust Reference: module privacy rules -- submodules can access parent module's private items via `super::`. Verified against Rust 2024 edition behavior.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH -- pure Rust module refactoring, no new dependencies
- Architecture: HIGH -- following established odb_core.rs precedent exactly
- Pitfalls: HIGH -- compiler enforces correctness; extraction is mechanical

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable -- no external dependencies)

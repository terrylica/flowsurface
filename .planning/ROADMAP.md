# Roadmap: Flowsurface Refactoring -- Maintainability Push

## Overview

Systematic refactoring of the flowsurface codebase targeting the top structural issues identified in the 4-agent audit: scattered config reads, two god modules (pane.rs at 2425 LOC, kline/mod.rs at 2388 LOC), bool flag arguments, and 36-file indicator ceremony. Each phase delivers one coherent structural improvement, verified by `cargo clippy -- -D warnings` and manual smoke testing. Config centralization first (lowest risk, unlocks everything), then safety annotations, then the two god module splits in dependency order, then cross-cutting indicator ceremony reduction.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Config Centralization** - Single AppConfig struct replaces scattered env var reads across 6 files
- [x] **Phase 2: Must-Use Safety Net** - #[must_use] annotations on Task/Effect return types before any code moves (completed 2026-03-28)
- [ ] **Phase 3: Bool-to-Enum Cleanup** - Replace 5 bool flag arguments with descriptive enums
- [ ] **Phase 4: Pane Content Extraction** - Extract Content enum and factory methods from pane.rs to pane/content.rs
- [ ] **Phase 5: Pane Stream Setup Extraction** - Extract stream wiring logic from pane.rs to pane/stream_setup.rs
- [ ] **Phase 6: Kline Data Ops Extraction** - Extract data operation methods from kline/mod.rs to kline/data_ops.rs
- [ ] **Phase 7: Kline ODB Lifecycle Extraction** - Extract ODB orchestration from kline/mod.rs to kline/odb_lifecycle.rs
- [ ] **Phase 8: Indicator Ceremony Reduction** - Reduce indicator addition touch points from 36 files to 6 or fewer

## Phase Details

### Phase 1: Config Centralization

**Goal**: All runtime configuration flows through a single validated struct -- no more hunting through 6 files for env var reads
**Depends on**: Nothing (first phase)
**Requirements**: CFG-01, CFG-02, CFG-03, CFG-04, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. A single `AppConfig` struct exists with all env vars, initialized via `LazyLock`
2. `grep -r 'std::env::var' src/ exchange/src/` returns zero hits outside the config module
3. App starts correctly with default config (no env vars set) and with full `.mise.toml` config
4. `cargo clippy -- -D warnings` passes clean
   **Plans:** 2 plans

Plans:

- [x] 01-01-PLAN.md -- Create AppConfig struct + migrate exchange crate (clickhouse.rs, telegram.rs)
- [x] 01-02-PLAN.md -- Migrate main crate (main.rs, logger.rs) + final verification

### Phase 2: Must-Use Safety Net

**Goal**: Compiler warns when Task/Effect/Action return values are silently dropped -- safety net before moving code between modules
**Depends on**: Phase 1
**Requirements**: QUAL-02, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. All `Action`, `Effect`, and `Task`-returning methods in pane.rs and dashboard.rs have `#[must_use]` annotations
2. `cargo clippy -- -D warnings` passes clean (no new unused-result warnings means existing code already handles returns correctly)
3. Intentionally dropping a Task return value in test code triggers a compiler warning
   **Plans:** 1/1 plans complete

Plans:

- [x] 02-01-PLAN.md -- Annotate 7 Action/Effect enums + 8 Option-returning functions with #[must_use]

### Phase 3: Bool-to-Enum Cleanup

**Goal**: Bool flag arguments replaced with self-documenting enums -- call sites read like prose instead of `true/false` mystery flags
**Depends on**: Phase 1
**Requirements**: QUAL-01, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. All 5 identified bool flag functions (adapter.rs, conditional_ema.rs, heatmap.rs, odb_core.rs, ladder.rs) use enum parameters instead
2. Every call site reads as `FetchMode::Initial` or `HeatmapLayer::Foreground` instead of bare `true`/`false`
3. `cargo clippy -- -D warnings` passes clean
   **Plans**: TBD

### Phase 4: Pane Content Extraction

**Goal**: Content enum and its factory methods live in their own file -- pane.rs loses ~500 LOC and Content becomes independently navigable
**Depends on**: Phase 2
**Requirements**: PANE-01, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. `src/screen/dashboard/pane/content.rs` exists with Content enum, all factory methods (`new_kline`, `new_heatmap`, etc.), and helper methods
2. `pane/mod.rs` re-exports Content via `pub(crate) use content::Content` -- all external imports unchanged
3. Opening any pane type (Kline, Heatmap, ODB) works identically to before
4. `cargo clippy -- -D warnings` passes clean
   **Plans**: TBD

### Phase 5: Pane Stream Setup Extraction

**Goal**: Stream wiring logic (resolve_content, set_content_and_streams) extracted -- pane.rs drops below 1500 LOC and stream setup is independently reviewable
**Depends on**: Phase 4
**Requirements**: PANE-02, PANE-03, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. `src/screen/dashboard/pane/stream_setup.rs` exists with `resolve_content()`, `set_content_and_streams()`, and `by_basis_default()`
2. ODB panes still register all 3 required streams (OdbKline + Trades + Depth) -- verified by opening an ODB pane
3. Switching tickers on a live pane refreshes streams correctly
4. `pane/mod.rs` is below 1500 LOC (measured by `wc -l`)
5. `cargo clippy -- -D warnings` passes clean
   **Plans**: TBD

### Phase 6: Kline Data Ops Extraction

**Goal**: Data insertion methods (insert_hist_klines, insert_open_interest, insert_trades) extracted from kline/mod.rs -- data flow paths are independently navigable
**Depends on**: Phase 4
**Requirements**: KLINE-01, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. `src/chart/kline/data_ops.rs` exists with all data insertion and aggregation query methods
2. Historical kline loading works (open pane -> bars appear)
3. Live trade insertion works (new trades flow into forming bar)
4. `cargo clippy -- -D warnings` passes clean
   **Plans**: TBD

### Phase 7: Kline ODB Lifecycle Extraction

**Goal**: ODB orchestration (gap-fill, reconciliation, sentinel, watchdog) extracted from kline/mod.rs -- fork-specific ODB complexity isolated in one file
**Depends on**: Phase 6
**Requirements**: KLINE-02, KLINE-03, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. `src/chart/kline/odb_lifecycle.rs` exists with ODB processor management, gap-fill fence, SSE reconciliation, sentinel audit, and watchdog methods
2. ODB pane loads historical bars from ClickHouse, gap-fills missing trades, and displays live forming bar
3. kline/mod.rs is below 1800 LOC (measured by `wc -l`)
4. No new RefCell borrow sites introduced -- all existing borrow discipline preserved
5. `cargo clippy -- -D warnings` passes clean
   **Plans**: TBD

### Phase 8: Indicator Ceremony Reduction

**Goal**: Adding a new indicator requires touching 6 or fewer files instead of 36 -- the highest-leverage change for future development velocity
**Depends on**: Phase 6, Phase 7
**Requirements**: QUAL-03, VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):

1. A documented checklist exists showing exactly which files to touch for a new indicator (6 or fewer)
2. The indicator registration path uses explicit match arms (not trait objects or macros) -- compiler-checked, not magic
3. `FOR_SPOT` and `FOR_PERPS` arrays remain as explicit lists
4. `cargo clippy -- -D warnings` passes clean
   **Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase                             | Plans Complete | Status      | Completed  |
| --------------------------------- | -------------- | ----------- | ---------- |
| 1. Config Centralization          | 2/2            | Complete    | 2026-03-27 |
| 2. Must-Use Safety Net            | 1/1 | Complete   | 2026-03-28 |
| 3. Bool-to-Enum Cleanup           | 0/TBD          | Not started | -          |
| 4. Pane Content Extraction        | 0/TBD          | Not started | -          |
| 5. Pane Stream Setup Extraction   | 0/TBD          | Not started | -          |
| 6. Kline Data Ops Extraction      | 0/TBD          | Not started | -          |
| 7. Kline ODB Lifecycle Extraction | 0/TBD          | Not started | -          |
| 8. Indicator Ceremony Reduction   | 0/TBD          | Not started | -          |

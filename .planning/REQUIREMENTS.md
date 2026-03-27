# Requirements: Flowsurface Refactoring -- Maintainability Push

**Defined:** 2026-03-27
**Core Value:** Every feature change should touch the minimum number of files necessary

## v1 Requirements

Requirements for this refactoring milestone. Each maps to roadmap phases.

### Config Centralization

- [ ] **CFG-01**: All env var reads centralized in a single `AppConfig` struct with `LazyLock` initialization
- [ ] **CFG-02**: Duplicate reads eliminated -- CH_HOST, CH_PORT, SSE_HOST, SSE_PORT each read exactly once
- [ ] **CFG-03**: Config validated eagerly at startup -- missing or invalid vars produce clear error messages before any network calls
- [ ] **CFG-04**: clickhouse.rs and telegram.rs import config values from the shared struct, not from `std::env::var` directly

### God Module Splits -- pane.rs

- [ ] **PANE-01**: Content type and its factory methods extracted to `pane/content.rs` (~500 LOC reduction)
- [ ] **PANE-02**: Stream setup logic (OdbKline + Trades + Depth wiring) extracted to `pane/stream_setup.rs` (~700 LOC reduction)
- [ ] **PANE-03**: pane.rs reduced below 1500 LOC after extractions while preserving all existing behavior

### God Module Splits -- kline/mod.rs

- [ ] **KLINE-01**: Data operation methods (TickAggr access, aggregation queries) extracted to `kline/data_ops.rs` (~200 LOC reduction)
- [ ] **KLINE-02**: ODB lifecycle orchestration (gap-fill, reconciliation, sidecar) extracted to `kline/odb_lifecycle.rs` (~600 LOC reduction)
- [ ] **KLINE-03**: kline/mod.rs reduced below 1800 LOC after extractions while preserving all canvas rendering behavior

### Code Quality

- [ ] **QUAL-01**: 5 bool flag arguments replaced with enums or split into separate functions (adapter.rs, conditional_ema.rs, heatmap.rs, odb_core.rs, ladder.rs)
- [ ] **QUAL-02**: `#[must_use]` added to Task/Effect return types in pane.rs and dashboard.rs before any code is moved
- [ ] **QUAL-03**: Indicator addition ceremony reduced from 36 file touch points to 6 or fewer

### Verification

- [ ] **VER-01**: `cargo clippy -- -D warnings` passes after every phase
- [ ] **VER-02**: Zero behavior changes -- all existing ODB, charting, and exchange functionality works identically
- [ ] **VER-03**: No new `unsafe` code introduced

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Additional God Module Splits

- **DASH-01**: dashboard.rs (1906 LOC) split into message dispatch + subscription modules
- **MAIN-01**: main.rs (1481 LOC) split into app config + modal orchestration

### Settings Decoupling

- **SET-01**: settings.rs decoupled from deep data model imports via config translation layer
- **SET-02**: comparison.rs (1844 LOC) exchange type dependency isolated via domain bridge

### Exchange Adapter Consolidation

- **ADAPT-01**: Shared WS/REST patterns extracted into `adapter/base.rs`
- **ADAPT-02**: Error handling unified across 5 exchange adapters

### Temporal Coupling

- **TEMP-01**: OnceLock statics (ODB_SYMBOLS, RUNTIME_PROXY_CFG, WRITER) replaced with explicit initialization assertions

## Out of Scope

| Feature                                                 | Reason                                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Crate-level splits                                      | Modules are cheaper; crate splits add build complexity                         |
| Exchange adapter trait abstraction                      | Rule of Three -- 5 adapters with unique protocol quirks, premature abstraction |
| New features                                            | Zero new functionality, pure structural improvement                            |
| UI/UX changes                                           | No visual changes whatsoever                                                   |
| Upstream compatibility shims                            | This is a fork with manual upstream merges                                     |
| DI framework                                            | Rust modules are the DI; frameworks are overkill                               |
| Proc macro config crates                                | Overkill for ~10 env vars; simple LazyLock struct is sufficient                |
| Layer-based file splitting (state.rs/view.rs/update.rs) | iced maintainer explicitly warns against this pattern                          |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase                      | Status  |
| ----------- | -------------------------- | ------- |
| CFG-01      | Phase 1                    | Pending |
| CFG-02      | Phase 1                    | Pending |
| CFG-03      | Phase 1                    | Pending |
| CFG-04      | Phase 1                    | Pending |
| PANE-01     | Phase 4                    | Pending |
| PANE-02     | Phase 5                    | Pending |
| PANE-03     | Phase 5                    | Pending |
| KLINE-01    | Phase 6                    | Pending |
| KLINE-02    | Phase 7                    | Pending |
| KLINE-03    | Phase 7                    | Pending |
| QUAL-01     | Phase 3                    | Pending |
| QUAL-02     | Phase 2                    | Pending |
| QUAL-03     | Phase 8                    | Pending |
| VER-01      | Phases 1-8 (cross-cutting) | Pending |
| VER-02      | Phases 1-8 (cross-cutting) | Pending |
| VER-03      | Phases 1-8 (cross-cutting) | Pending |

**Coverage:**

- v1 requirements: 16 total
- Mapped to phases: 16/16
- Unmapped: 0

---

_Requirements defined: 2026-03-27_
_Last updated: 2026-03-27 after roadmap creation_

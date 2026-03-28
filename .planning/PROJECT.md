# Flowsurface Refactoring — Maintainability Push

## What This Is

A systematic refactoring of the flowsurface codebase to improve maintainability, guided by the refactoring guide's detection heuristics. Targets the top 5 worst offenders identified in a comprehensive 4-agent audit: scattered config, god modules (pane.rs, kline/mod.rs), exchange adapter duplication, and settings coupling.

## Core Value

Every feature change should touch the minimum number of files necessary — no shotgun surgery, no god modules, no duplicated config reads.

## Requirements

### Validated

- ✓ Codebase compiles clean with `clippy -D warnings` — existing
- ✓ ODB triple-stream architecture works end-to-end — existing
- ✓ ClickHouse adapter handles HTTP + SSE + catchup — existing
- ✓ 5 exchange adapters (Binance, Bybit, OKX, Hyperliquid, ClickHouse) — existing
- ✓ Chart rendering with 4-layer canvas (main, watermark, legend, crosshair) — existing
- ✓ Indicator system with 6 KlineIndicator types — existing
- ✓ SSH tunnel reliability with autossh + retry backoff — existing
- ✓ Centralized AppConfig struct replaces scattered env var reads — Phase 1
- ✓ `#[must_use]` on all Action/Effect return types — safety net for code moves — Phase 2

### Active

- [x] Centralize all env var reads into a single config struct
- [ ] Split pane.rs (2425 LOC) into event dispatch + chart factory + interaction modules
- [ ] Split kline/mod.rs (2388 LOC) into canvas rendering + data operations modules
- [ ] Reduce indicator addition ceremony from 36 files to fewer touch points
- [ ] Eliminate duplicate env var reads across clickhouse.rs and telegram.rs
- [ ] Extract shared exchange adapter patterns (WS connect, REST, error handling)
- [ ] Decouple settings.rs from deep data model imports

### Out of Scope

- Upstream compatibility shims — we fork, not merge-friendly refactoring
- New features — zero new functionality, pure structural improvement
- Crate-level splits — not splitting workspace into more crates
- Exchange adapter trait abstraction — too much indirection for 5 adapters
- UI/UX changes — no visual changes whatsoever

## Context

- **Codebase**: Rust + iced 0.14 + WGPU, ~118 .rs files across 3 workspace crates
- **Fork of**: flowsurface-rs/flowsurface with ODB (Open Deviation Bar) additions
- **Audit findings**: 8 god modules (>500 LOC), shotgun surgery on indicator adds (36 files), scattered env reads in 6 files, feature envy in 5 files
- **Codebase map**: `.planning/codebase/` (7 documents, 2223 lines total)
- **Refactoring guide**: `/quality-tools:refactoring-guide` — detection heuristics, structural coupling, type design, architecture, module boundaries, tactical moves, Rust-specific guidance

### Audit Summary (from 4-agent scan, 2026-03-27)

| Issue              | Severity | Evidence                                                          |
| ------------------ | -------- | ----------------------------------------------------------------- |
| God modules        | HIGH     | 8 files >500 LOC, worst: pane.rs (2425), kline/mod.rs (2388)      |
| Shotgun surgery    | HIGH     | RSI indicator commit touched 36 files                             |
| Scattered env vars | HIGH     | 6 files, 4 duplicate reads (CH_HOST, CH_PORT, SSE_HOST, SSE_PORT) |
| Feature envy       | MEDIUM   | 5 files with 8-26 cross-module imports                            |
| Bool flag args     | MEDIUM   | 5 public functions with branching bool params                     |
| Temporal coupling  | MEDIUM   | 3 OnceLock statics with silent fallback on init failure           |
| Mixed concerns     | MEDIUM   | query() mixes data fetch + Telegram alerting + retry logic        |
| Import cycles      | CLEAN    | No circular deps                                                  |
| Dead code          | CLEAN    | No stale TODOs, legitimate #[allow(dead_code)]                    |

## Constraints

- **No regressions**: Must compile clean (`cargo clippy -- -D warnings`) after every phase
- **No behavior change**: All existing functionality must work identically
- **Incremental**: Each phase is independently shippable — no big-bang rewrites
- **Rust edition**: 2024, toolchain 1.93.1

## Key Decisions

| Decision                                | Rationale                                                                     | Outcome        |
| --------------------------------------- | ----------------------------------------------------------------------------- | -------------- |
| Worst offenders first, not full repo    | Maximize impact per effort; full repo would be months                         | v1.0 Validated |
| Config struct over dependency injection | Rust LazyLock pattern is idiomatic; no need for DI framework                  | v1.0 Validated |
| Module splits over crate splits         | Modules are cheaper to refactor; crate splits add build complexity            | v1.0 Validated |
| No upstream compatibility constraint    | This is a fork with divergent ODB features; upstream merges are manual anyway | v1.0 Validated |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):

1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):

1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---

## Current State

**v1.0 Milestone shipped 2026-03-28.** All 16 requirements satisfied. God modules tamed: pane.rs (2431→1409), kline/mod.rs (2388→1721). Indicator ceremony: 3 files. Config centralized. Bool flags eliminated. Safety net in place.

**Next milestone candidates** (from v2 requirements in archived REQUIREMENTS.md):

- DASH-01: dashboard.rs split (1906 LOC)
- MAIN-01: main.rs split (1481 LOC)
- SET-01/SET-02: settings decoupling
- ADAPT-01/ADAPT-02: exchange adapter consolidation

_Last updated: 2026-03-28 after v1.0 milestone completion_

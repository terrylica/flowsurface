# Feature Landscape: Rust + iced Codebase Refactoring

**Domain:** Structural refactoring of a large Rust GUI application (iced 0.14, Elm architecture)
**Researched:** 2026-03-26
**Overall confidence:** HIGH (evidence from codebase audit + established refactoring patterns + iced community guidance)

## Table Stakes

Features users (developers maintaining this codebase) expect. Missing = refactoring feels incomplete.

| Feature                                       | Why Expected                                                                                                                                                                                 | Complexity | Notes                                                                                                                                                                                |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Config centralization (env vars)              | 6 files with scattered `std::env::var` reads, 4 duplicated (CH_HOST, CH_PORT, SSE_HOST, SSE_PORT). Every env var change requires grep-and-pray.                                              | Low        | Single `AppConfig` struct with `LazyLock<AppConfig>`. Pure extraction, no behavior change.                                                                                           |
| God module splitting: pane.rs (2425 LOC)      | Combines event dispatch, chart factory, stream resolution, pane lifecycle, UI rendering. Cognitive load is unsustainable.                                                                    | Medium     | Split by concern: event dispatch, content resolution, view rendering. Keep types in parent mod.                                                                                      |
| God module splitting: kline/mod.rs (2388 LOC) | Already has 4 submodules extracted (bar_selection, crosshair, odb_core, rendering) but mod.rs still owns canvas Program impl, data operations, gap-fill orchestration, indicator management. | Medium     | Extract data operations (insert_trades, kline management) and gap-fill orchestration into submodules. Canvas Program impl stays in mod.rs as coordinator.                            |
| Reduce indicator addition ceremony            | RSI commit touched 36 files. Adding an indicator requires: enum variant, Display impl, FOR_SPOT array, FOR_PERPS array, factory registration, renderer file, settings UI integration.        | Medium     | Inventory-based registration: derive macro or const array that auto-generates Display + array membership. Factory uses match on enum, unavoidable in Rust. Target: 4-6 touch points. |
| Eliminate duplicate env var reads             | CH_HOST read in clickhouse.rs (line 239) AND telegram.rs (line 114). Same for CH_PORT, SSE_HOST, SSE_PORT. Defaults differ between sites ("bigblack" vs "localhost").                        | Low        | Direct consequence of config centralization. Both modules import from single config.                                                                                                 |

## Differentiators

Techniques that go beyond minimum viable refactoring. Not expected, but significantly improve long-term maintainability.

| Feature                                                 | Value Proposition                                                                                                                                            | Complexity | Notes                                                                                                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Newtype wrappers for stringly-typed config              | `ChHost(String)`, `ChPort(u16)` instead of raw strings. Prevents passing SSE host where CH host is expected.                                                 | Low        | Rust type system catches misuse at compile time. Small effort, high payoff.                                                                           |
| Feature-oriented module boundaries (iced pattern)       | Instead of splitting by technical layer (state/view/update), split by feature domain. iced maintainer explicitly recommends this over layer-based splitting. | Medium     | pane.rs splits into: pane/lifecycle.rs, pane/odb.rs, pane/streaming.rs. Each owns its Message variants, update logic, and view fragments.             |
| OnceLock/LazyLock audit with explicit error propagation | 3 OnceLock statics silently fall back on init failure (temporal coupling). Replace with `Result`-returning init or panic-on-misconfiguration.                | Low        | Fail-fast principle. Silent fallback masks deployment errors.                                                                                         |
| Bool parameter elimination                              | 5 public functions use bool flags for branching. Replace with enum parameters: `enum FetchMode { Initial, Pagination }` instead of `is_initial: bool`.       | Low        | Self-documenting call sites. `fetch(FetchMode::Initial)` vs `fetch(true)`.                                                                            |
| Shared exchange adapter patterns                        | WS connect, REST retry, error handling duplicated across 5 adapters (Binance, Bybit, OKX, Hyperliquid, ClickHouse).                                          | High       | Extract `trait ExchangeTransport` or shared helper functions. PROJECT.md marks full trait abstraction as out-of-scope; helper extraction is in-scope. |
| ClickHouse adapter decomposition                        | 1419 LOC monolith. SQL builder, HTTP polling, SSE streaming, gap-fill catchup all interleaved.                                                               | Medium     | Extract: `sql.rs` (pure functions), `sse.rs` (stream logic), `catchup.rs` (gap-fill). HTTP client stays in mod.rs.                                    |
| Message enum namespacing                                | pane.rs `Message` enum has 20+ variants mixing pane lifecycle, chart interaction, and settings.                                                              | Medium     | Group into sub-enums: `PaneMessage::Layout(LayoutMsg)`, `PaneMessage::Chart(ChartMsg)`. Requires updating match arms but improves readability.        |

## Anti-Features

Things to deliberately NOT do during this refactoring.

| Anti-Feature                                                | Why Avoid                                                                                                                                                                                           | What to Do Instead                                                                                                                                  |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Crate-level splits                                          | PROJECT.md explicitly marks this out of scope. Adds build complexity (longer compile times, workspace management) without proportional benefit for 118 files.                                       | Use module splits within existing crates. `mod foo { mod bar; }` achieves same encapsulation without build overhead.                                |
| Layer-based file splitting (state.rs / view.rs / update.rs) | iced maintainer explicitly warns against this. State, message, update, and view are tightly coupled in Elm architecture. Separating them creates artificial boundaries and makes navigation harder. | Split by feature/domain instead. A pane module owns its state + message + update + view together.                                                   |
| Trait-heavy abstraction for exchange adapters               | 5 adapters with different APIs (REST, WS, HTTP polling, SSE). Forcing them behind a shared trait creates a lowest-common-denominator interface. PROJECT.md marks this out of scope.                 | Extract shared utility functions (reconnect backoff, HTTP retry, WS timeout) without forcing a common trait.                                        |
| Macro-heavy code generation for indicators                  | Proc macros are hard to debug, slow compile times, and opaque to IDE tooling. The indicator ceremony (36 files) is painful but mostly grep-able.                                                    | Use declarative const arrays and a registration pattern. Reduce touch points from 36 to 4-6 through consolidation, not generation.                  |
| Global dependency injection framework                       | Rust has no runtime DI. Attempts to simulate it (trait objects, Arc<dyn>) add indirection without benefit. Config is the only "global" need.                                                        | `LazyLock<AppConfig>` for config. Direct imports for everything else. Rust's module system IS the DI framework.                                     |
| Refactoring upstream-owned code paths                       | pane.rs and kline/mod.rs both have `FILE-SIZE-OK: upstream file` comments. Deep restructuring diverges from upstream, making future merges painful.                                                 | Surgical extractions only. Move fork-specific code into new submodules. Leave upstream match arms in place. Minimize diff surface against upstream. |
| Big-bang restructuring                                      | Attempting all refactoring in one phase risks regressions. The codebase has no integration test coverage for ODB reconciliation, pane streams, or gap-fill.                                         | Incremental phases. Each phase is independently shippable. Clippy + manual smoke test after each.                                                   |

## Feature Dependencies

```
Config Centralization
  |
  +---> Eliminate Duplicate Env Reads (direct consequence)
  +---> OnceLock Audit (config init is primary OnceLock use case)
  +---> Newtype Wrappers (applied to config fields)

God Module Splitting (pane.rs)
  |
  +---> Message Enum Namespacing (sub-enums live in split submodules)
  +---> Feature-Oriented Boundaries (splitting strategy)

God Module Splitting (kline/mod.rs)
  |
  +---> ClickHouse Adapter Decomposition (data ops reference CH types)

Indicator Ceremony Reduction
  (independent — can be done in any order)

Bool Parameter Elimination
  (independent — can be done in any order)

Exchange Adapter Patterns
  |
  +---> ClickHouse Adapter Decomposition (CH adapter is one of the 5)
```

## MVP Recommendation

Prioritize (Phase 1 -- highest impact, lowest risk):

1. **Config centralization** -- Low complexity, eliminates 4 duplicate reads, unlocks newtype wrappers and OnceLock audit. Every subsequent refactoring benefits from a single config source.
2. **Bool parameter elimination** -- Low complexity, independent, immediate readability win. Good warm-up for larger splits.
3. **God module split: pane.rs** -- Medium complexity, but this is the #1 pain point (2425 LOC, event dispatch mixed with rendering). Split into lifecycle/resolution/view submodules.

Defer to Phase 2:

- **kline/mod.rs split** -- Already has 4 submodules extracted. Remaining work is medium-complexity data operation extraction. Less urgent than pane.rs.
- **Indicator ceremony reduction** -- The 36-file problem is painful but rare (new indicators are added infrequently). Lower urgency than daily-touched god modules.

Defer to Phase 3:

- **ClickHouse adapter decomposition** -- 1419 LOC but self-contained. Low daily pain because it is a single adapter, not a cross-cutting concern.
- **Exchange adapter shared patterns** -- High complexity, moderate payoff. The 5 adapters work; duplication is tolerable until a 6th adapter is added.

## Sources

- [iced Discussion #1572: Module splitting patterns](https://github.com/iced-rs/iced/discussions/1572) -- iced maintainer recommends feature-based, not layer-based splitting
- [Elm Architecture scaling patterns](https://elm-radio.com/episode/scaling-elm-apps/) -- subdivide only what is unwieldy, keep state/update/view together
- [Rust LazyLock documentation](https://dev-doc.rust-lang.org/nightly/std/sync/struct.LazyLock.html) -- standard library config pattern
- [Shotgun surgery refactoring](https://refactoring.guru/smells/shotgun-surgery) -- move related logic to single class/module
- [Rust project structure best practices](https://www.slingacademy.com/article/best-practices-for-structuring-large-scale-rust-applications-with-modules/) -- domain-based modularization
- Codebase audit: `.planning/codebase/CONCERNS.md` (2026-03-26) -- severity ratings and evidence
- Codebase audit: `.planning/PROJECT.md` (2026-03-27) -- scope constraints and decisions

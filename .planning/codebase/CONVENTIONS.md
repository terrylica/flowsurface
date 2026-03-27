# Coding Conventions

**Analysis Date:** 2026-03-26

## Naming Patterns

**Files:**

- Lowercase with underscores: `kline.rs`, `clickhouse.rs`, `bar_selection.rs`
- Submodules grouped in directories: `src/chart/`, `src/modal/`, `src/connector/`
- Monolithic files with `FILE-SIZE-OK` header comment if logically too coupled to split (e.g., `src/chart/kline/mod.rs`, `exchange/src/adapter/clickhouse.rs`)

**Functions:**

- Snake case: `fetch_klines()`, `guard_allows()`, `build_odb_sql()`
- Getter/setter pattern: `fn state(&self)`, `fn mut_state(&mut self)`, not `get_state()`
- Helpers with clear intent: `is_gap()`, `is_full_range_bar()`, `should_show_text()`

**Variables:**

- Snake case: `prev_bar`, `cell_width`, `visible_region`
- Abbreviations allowed in numeric contexts: `k_actual`, `dbps`, `bps`, `ms`, `us` (milliseconds, microseconds)
- Option/Result fields often descriptive: `through_agg_id`, `agg_trade_id_range`, `open_time_ms`

**Types:**

- PascalCase for structs/enums: `OpenDeviationBar`, `KlineIndicator`, `SessionBoundary`, `ChMicrostructure`
- Enum variants: `PascalCase` with semantic grouping (e.g., `StreamKind::Kline`, `ContentKind::OdbChart`)
- Type aliases for common patterns: `type BufferedChKline = (...)` — documents intent through naming

**Constants:**

- UPPERCASE with underscores: `ODB_THRESHOLDS`, `NY_OPEN_20260302_MS`, `WS_READ_TIMEOUT`
- Grouped in structs or modules when related: `const KLINE: [Timeframe; 10] = [...]`

## Code Style

**Formatting:**

- Max width: **100 characters** (configured in `rustfmt.toml`)
- Edition: `2024` (configured in `Cargo.toml` workspace)
- Indent: 4 spaces (Rust default)

**Linting:**

- Tool: `clippy` with `-D warnings` (deny warnings as errors)
- Config: `clippy.toml` sets `too-many-arguments-threshold = 16` and `enum-variant-name-threshold = 5`
- Run: `mise run lint` → `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings`

## Import Organization

**Order:**

1. Standard library imports (`use std::...`)
2. External crates in dependency order (`use iced::...`, `use serde::...`, `use tokio::...`)
3. Internal crates (`use exchange::...`, `use data::...`)
4. Module-relative imports (`use super::...`, `use crate::...`)
5. Macro imports (`use enum_map::enum_map;`)

**Path Aliases:**

- No path aliases configured
- Relative imports use `super::` for siblings, `crate::` for crate root
- Full module paths used for clarity: `src/chart/indicator/kline/` → `crate::chart::indicator::kline`

**Re-exports:**

- Barrel files used to simplify imports: `pub use interaction::{...}` in `src/chart.rs`
- Downstream code can use `use super::*` and `use crate::chart::{...}` without breaking on refactors

## Error Handling

**Patterns:**

- Custom error enums with `#[derive(thiserror::Error)]`: `AudioError`, `ReqError`, `DashboardError`
- `Result<T, E>` return type standard for fallible operations
- Early return with `?` operator for error propagation
- Explicit `unwrap()` only in tests or panic-appropriate contexts
- Guard patterns with boolean checks before fallible operations

**Examples:**

- `fn new(volume: Option<f32>) -> Result<Self, AudioError>` — explicit error type
- `Response::to_result()` — conversion to expected type at boundary
- Tokio runtime context requirement: `let rt = tokio::runtime::Runtime::new().unwrap(); let _guard = rt.enter();` — test helper for async code

**Error context:**

- Errors logged with `log::error!()` or `tg_alert!(critical/warning)` for telemetry
- User-facing errors propagated through `Message` enum in UI
- Network errors captured in `ReqError` and `AdapterError` enums

## Logging

**Framework:** `log` crate (not `tracing`)

**Patterns:**

- `log::trace!()` — detailed diagnostics (e.g., chart rendering loops, boundary calculations)
- `log::info!()` — startup state, feature toggles, significant events (e.g., SSE enabled/disabled)
- `log::warn!()` — recoverable issues (e.g., partial fetch results)
- `log::error!()` — errors that degrade functionality

**Log output:**

- File: `~/Library/Logs/flowsurface.log` (configured in `src/logger.rs` via fern)
- Rotation: on-demand via `fern::log_file()` without abort on rotation failure
- Levels: Debug mode defaults to `Level::Debug`, Release defaults to `Level::Info`

**Telemetry integration:**

- Critical issues sent via `tg_alert!(critical: "...")` macro (49 alert sites across codebase)
- Cooldown: 30-second window (`should_alert()`) prevents spam for same-category alerts
- Telegram feature: `FLOWSURFACE_TG_BOT_TOKEN` + `FLOWSURFACE_TG_CHAT_ID` env vars

## Comments

**When to Comment:**

- Explain _why_, not _what_: "ODB bars require explicit open_time_ms threading; Kline.time is close_time_ms" not "get open_time_ms"
- Document non-obvious trade-offs: "// Dedup fence: WS trades with id <= fence are skipped; ensures no duplication after gap-fill"
- Link to GitHub issues: `// GitHub Issue: https://github.com/terrylica/flowsurface/issues/100`
- Highlight fork-specific code: `// NOTE(fork):` prefix for changes from upstream

**JSDoc/TSDoc (Rust doc comments):**

- `///` for public API documentation (functions, types, constants)
- `//!` for module-level documentation (rarely used)
- Doc comments required on public exports; internal functions use `//` comments
- Example from `src/adapter/clickhouse.rs`:

  ```rust
  /// Convert a flowsurface Trade into an opendeviationbar-core AggTrade.
  ///
  /// Both Price and FixedPoint use i64 with 10^8 scale, so price conversion
  /// is a direct copy of the underlying units. Volume uses f32→FixedPoint
  /// via string round-trip for precision.
  pub fn trade_to_agg_trade(trade: &Trade, seq_id: i64) -> ...
  ```

## Function Design

**Size:** Functions typically 20–80 lines; helpers factored into sub-functions when approaching 120 lines (see `src/chart/kline/mod.rs` where rendering logic split into `rendering.rs` module)

**Parameters:**

- Prefer explicit parameters over `self` capture when intent is clearer
- Use `&T` for read-only access, `&mut T` for mutable
- Generic bounds documented with examples: `impl Chart: PlotConstants + canvas::Program<Message>`
- Builder pattern used for complex configurations: `OdbSseConfig::new().with_timeout(...)`

**Return Values:**

- Functions return owned types (`Vec<T>`, `String`) when modifying data
- Borrowed references (`&[T]`, `&str`) returned when non-owning view needed
- Option/Result used consistently: `Option<(u64, u64)>` for optional ranges, `Result<T, E>` for errors
- `None` sentinels used for special values: `u64::MAX` in `FetchRange::Kline(0, u64::MAX)` signals full-reload mode

## Module Design

**Exports:**

- `pub mod` for submodules, `pub use` for re-exported types
- Private modules (`mod interaction;`) for tightly-coupled logic
- Example from `src/chart.rs`:

  ```rust
  pub mod comparison;
  pub(crate) mod interaction;    // Private to crate
  pub use interaction::{AxisScaleClicked, Interaction, Message};
  pub(crate) use legend::draw_volume_bar;  // Re-export for internal use
  ```

**Barrel Files:**

- `src/chart.rs` acts as facade for submodules, re-exporting public types
- Allows downstream to write `use crate::chart::{ViewState, Message}` without path depth
- Maintained when refactoring to avoid breaking public API

**Module Visibility:**

- `pub mod` — public submodules exported from crate root
- `pub(crate)` — visible throughout crate, not outside
- Unmarked — private to module
- `pub(super)` — visible to parent module only

**Workspace Organization:**
Three crates with clear boundaries:

- `flowsurface` — GUI, chart rendering, event handling (`src/`)
- `flowsurface-exchange` (`exchange/`) — adapters, WebSocket, REST
- `flowsurface-data` (`data/`) — aggregation, models, indicators

No circular dependencies; data/exchange are pure logic crates, flowsurface integrates them.

## Fork-Specific Patterns

**GitHub Issue Links:**

- Every fork-specific file has a reference comment: `// GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/91`
- Helps upstream sync and identifies fork scope

**Fork Notes:**

- Comments prefixed with `// NOTE(fork):` mark deviations from upstream
- Example: `// NOTE(fork): issue#100 — keyboard chart navigation` in `src/chart/keyboard_nav.rs`
- Temporary workarounds flagged: `// FIXME: ...` with context

**Temporal Annotations:**

- `// FILE-SIZE-OK:` explains why monolithic files aren't split
- Example: `// FILE-SIZE-OK: monolithic adapter — CH HTTP, SSE, catchup, SQL builder are tightly coupled`
- Used to prevent future refactors breaking intentional design

---

_Convention analysis: 2026-03-26_

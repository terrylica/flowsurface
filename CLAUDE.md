# CLAUDE.md - Project Hub

**flowsurface**: Native desktop charting app for crypto markets. Rust + iced 0.14 + WGPU. This fork adds **ODB (Open Deviation Bar) visualization** from precomputed [opendeviationbar-py](https://github.com/terrylica/opendeviationbar-py) cache via ClickHouse.

**Upstream**: [flowsurface-rs/flowsurface](https://github.com/flowsurface-rs/flowsurface) | **Fork**: [terrylica/flowsurface](https://github.com/terrylica/flowsurface)

---

## Quick Reference

| Task                 | Command / Entry Point         | Details                                 |
| -------------------- | ----------------------------- | --------------------------------------- |
| Build & run          | `mise run run`                | Preflight + cargo run                   |
| Launch .app bundle   | `mise run run:app`            | Preflight + open Flowsurface.app        |
| ClickHouse preflight | `mise run preflight`          | Tunnel + connectivity + data validation |
| SSH tunnel           | `mise run tunnel:start`       | localhost:18123 → bigblack:8123         |
| Release build        | `mise run build:release`      | Optimized binary                        |
| .app bundle          | `mise run release:app-bundle` | Build + update .app + register icon     |
| Lint                 | `mise run lint`               | fmt:check + clippy                      |
| Sync upstream        | `mise run upstream:diff`      | Show new upstream commits               |

---

## CLAUDE.md Network (Hub-and-Spoke)

| Directory       | CLAUDE.md                                      | Scope                                                                   |
| --------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| `/`             | This file                                      | Hub — architecture, env vars, patterns, errors                          |
| `/exchange/`    | [exchange/CLAUDE.md](exchange/CLAUDE.md)       | Exchange adapters, ClickHouse, SSE, stream types                        |
| `/data/`        | [data/CLAUDE.md](data/CLAUDE.md)               | Chart types, indicators, aggregation, layout                            |
| `/docs/audits/` | [docs/audits/CLAUDE.md](docs/audits/CLAUDE.md) | Statistical audits: bar-selection metrics (v1 threshold, v2 rank-based) |

---

## Architecture

```
flowsurface/                 Main crate — GUI, chart rendering, event handling
├── exchange/                Exchange adapters, WebSocket/REST/HTTP streams
│   └── adapter/
│       ├── clickhouse.rs    ODB adapter (HTTP + SSE, reads opendeviationbar-py cache)
│       ├── binance.rs       Binance Spot + Perpetuals
│       ├── bybit.rs         Bybit Perpetuals
│       ├── hyperliquid.rs   Hyperliquid DEX
│       └── okex.rs          OKX Multi-product
├── data/                    Data aggregation, indicators, layout models
│   ├── chart.rs             Basis enum (Time, Tick, Odb)
│   ├── chart/indicator.rs   KlineIndicator enum (6 types)
│   ├── aggr/ticks.rs        TickAggr, RangeBarMicrostructure
│   └── session.rs           Trading session boundaries (NY/London/Tokyo)
└── src/                     GUI application
    ├── chart/kline.rs       Chart rendering (candles, ODB bars, footprint)
    ├── chart/indicator/     Indicator renderers (volume, delta, OFI, etc.)
    ├── chart/session.rs     Session line rendering
    ├── connector/           Stream connection + data fetching
    │   ├── stream.rs        ResolvedStream, stream matching
    │   └── fetcher.rs       FetchedData, RequestHandler, batch fetching
    ├── screen/dashboard/    Pane grid UI + pane state
    ├── modal/               Settings & configuration modals
    └── widget/              BTC widget overlay
```

---

## Environment Variables

All set in `.mise.toml`. The app reads them at runtime via `std::env::var()`.

| Variable                     | Default     | Purpose                             |
| ---------------------------- | ----------- | ----------------------------------- |
| `FLOWSURFACE_CH_HOST`        | `bigblack`  | ClickHouse HTTP host                |
| `FLOWSURFACE_CH_PORT`        | `8123`      | ClickHouse HTTP port                |
| `FLOWSURFACE_SSE_ENABLED`    | `false`     | Enable SSE live bar stream          |
| `FLOWSURFACE_SSE_HOST`       | `localhost` | SSE sidecar host                    |
| `FLOWSURFACE_SSE_PORT`       | `8081`      | SSE sidecar port                    |
| `FLOWSURFACE_OUROBOROS_MODE` | `day`       | ODB session mode (`day` or `month`) |
| `FLOWSURFACE_ALWAYS_ON_TOP`  | _(unset)_   | Pin window above all others if set  |
| `FLOWSURFACE_TG_BOT_TOKEN`   | _(unset)_   | Telegram bot token for alerts       |
| `FLOWSURFACE_TG_CHAT_ID`     | _(unset)_   | Telegram chat ID for alerts         |

---

## ODB Integration (Fork-Specific)

ODB panes use **triple-stream architecture**:

```
Stream 1: OdbKline — ClickHouse (completed bars, 5s poll)
  → fetch_klines() → ChKline → Kline → TickAggr
  → update_latest_kline() → replace_or_append_kline()

Stream 2: Trades — Binance @aggTrade WebSocket (live trades)
  → TradesReceived → insert_trades_buffer()
  → TickAggr::insert_trades() → is_full_range_bar(threshold_dbps)
  → Forming bar oscillates until threshold breach → bar completes

Stream 3: Depth — Binance depth WebSocket (orderbook)
  → DepthReceived → heatmap / footprint data

Reconciliation: ClickHouse bar replaces locally-built bar (authoritative)

Stream 4: Gap-fill — ODB sidecar Ariadne + /trades/gap-fill
  → After initial CH klines load, query Ariadne for last_agg_trade_id
  → Fetch missing trades via /trades/gap-fill (Parquet fast path)
  → Dedup fence: WS trades with id <= fence are skipped
  → CH bars buffered during gap-fill, flushed after completion
```

**CRITICAL**: ODB panes must subscribe to ALL THREE streams (`OdbKline`, `Trades`, `Depth`) in `resolve_content()` at `src/screen/dashboard/pane.rs`. Missing `Trades` causes "Waiting for trades..." forever because `matches_stream()` silently drops unmatched events.

**Key types** (see [data/CLAUDE.md](data/CLAUDE.md) and [exchange/CLAUDE.md](exchange/CLAUDE.md) for details):

| Type                     | Location                             | Purpose                                    |
| ------------------------ | ------------------------------------ | ------------------------------------------ |
| `Basis::Odb(u32)`        | `data/src/chart.rs`                  | Chart basis (threshold in dbps)            |
| `KlineChartKind::Odb`    | `data/src/chart/kline.rs`            | Chart type variant                         |
| `RangeBarMicrostructure` | `data/src/aggr/ticks.rs`             | Sidecar: trade_count, ofi, trade_intensity |
| `ChKline`                | `exchange/src/adapter/clickhouse.rs` | ClickHouse row deserialization             |
| `ODB_THRESHOLDS`         | `data/src/chart.rs`                  | `[100, 250, 500, 750]` dbps                |
| `ContentKind::OdbChart`  | `data/src/layout/pane.rs`            | Pane serialization variant                 |

**Threshold display**: `BPR{dbps/10}` — BPR25 = 250 dbps = 0.25%, BPR50 = 500 dbps, etc.

---

## ClickHouse Infrastructure

All range bar data served from **bigblack** via SSH tunnel. No local ClickHouse.

| Setting    | Value                   | Source       |
| ---------- | ----------------------- | ------------ |
| Host       | `localhost`             | `.mise.toml` |
| Port       | `18123`                 | `.mise.toml` |
| SSH tunnel | `18123 → bigblack:8123` | `infra.toml` |

**Preflight** (`mise run preflight`): Runs before `mise run run` and `mise run run:app`:

1. Establishes SSH tunnel (idempotent)
2. Verifies ClickHouse responds (3 retries)
3. Verifies `opendeviationbar_cache.open_deviation_bars` table exists
4. Verifies BTCUSDT data present for all thresholds

---

## Mise Tasks

### Dev (`.mise/tasks/dev.toml`)

| Task            | Description                   | Depends On  |
| --------------- | ----------------------------- | ----------- |
| `build`         | Debug binary                  | —           |
| `build:release` | Optimized release binary      | —           |
| `run`           | Build + run with ClickHouse   | `preflight` |
| `run:app`       | Launch Flowsurface.app bundle | `preflight` |
| `check`         | Type-check (no codegen)       | —           |
| `clippy`        | Lint with `-D warnings`       | —           |
| `fmt`           | Format all Rust code          | —           |
| `lint`          | `fmt:check` + `clippy`        | —           |

### Release (`.mise/tasks/release.toml`)

| Task                  | Description                                  |
| --------------------- | -------------------------------------------- |
| `release:macos`       | Universal binary (x86_64 + aarch64 via lipo) |
| `release:macos-arm64` | aarch64-only release                         |
| `release:app-bundle`  | Build + update .app + sign + register icon   |
| `sign:app`            | Ad-hoc codesign the .app bundle              |

### Infrastructure (`.mise/tasks/infra.toml`)

| Task            | Description                                   |
| --------------- | --------------------------------------------- |
| `tunnel:start`  | SSH tunnel to bigblack (idempotent)           |
| `tunnel:stop`   | Kill SSH tunnel                               |
| `tunnel:status` | Verify tunnel + ClickHouse connectivity       |
| `preflight`     | Full validation (tunnel + CH + schema + data) |

### Upstream (`.mise/tasks/upstream.toml`)

| Task              | Description               |
| ----------------- | ------------------------- |
| `upstream:fetch`  | Fetch upstream changes    |
| `upstream:diff`   | Show new upstream commits |
| `upstream:merge`  | Merge upstream/main       |
| `upstream:rebase` | Rebase onto upstream/main |

---

## Release Model

**Native desktop app** — no crates.io, no version tags, no changelog.

| Task                           | What It Does                                     |
| ------------------------------ | ------------------------------------------------ |
| `mise run build:release`       | Optimized binary at `target/release/flowsurface` |
| `mise run release:app-bundle`  | Build + update `.app` + SSH launcher + icon      |
| `mise run release:macos-arm64` | aarch64-only release binary                      |
| `mise run release:macos`       | Universal binary (x86_64 + aarch64 via lipo)     |

**Code signing**: Ad-hoc via `codesign --deep --force --sign -` (built into `run:app` and `release:app-bundle`).

---

## Common Patterns

### Adding a New Indicator

1. Add variant to `KlineIndicator` enum in `data/src/chart/indicator.rs`
2. Add to `FOR_SPOT` and/or `FOR_PERPS` arrays
3. Add `Display` impl
4. Create indicator file in `src/chart/indicator/kline/`
5. Implement `KlineIndicatorImpl` trait
6. Register in factory `src/chart/indicator/kline.rs`

### Extending ODB Support

When modifying ODB rendering or behavior, check **all** match arms for `Basis::Odb(_)`, `KlineChartKind::Odb`, and `ContentKind::OdbChart` across:

- `src/screen/dashboard/pane.rs` — pane streams (must include `OdbKline` + `Depth` + `Trades`)
- `src/screen/dashboard.rs` — event dispatch, pane switching
- `src/chart/kline.rs` — rendering, trade insertion
- `src/chart/heatmap.rs` — depth heatmap
- `src/modal/pane/stream.rs` — settings UI
- `src/modal/pane/settings.rs` — chart config
- `data/src/layout/pane.rs` — serialization

### iced Canvas Architecture

**Four geometry layers** in `src/chart/kline.rs` (stacked in draw order):

| Layer       | Frame transforms?             | When cleared            |
| ----------- | ----------------------------- | ----------------------- |
| `main`      | translate → scale → translate | Panning, zoom, new bars |
| `watermark` | None (screen-space)           | Rarely                  |
| `legend`    | None (screen-space)           | Every cursor move       |
| `crosshair` | None (screen-space)           | Every cursor move       |

**Chart-space → screen-space formula** (canonical, from `keyboard_nav.rs`):

```
screen_x = (chart_x + translation.x) * scaling + bounds.width / 2
```

ODB: `chart_x = -(visual_idx * cell_width)`. Lower `visual_idx` = newer bar = higher screen_x.

**Hit detection — anti-pattern vs correct pattern:**

| ❌ Anti-pattern                                                            | ✅ Correct pattern                                                                              |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Compute `screen_x` from formula; check `(cursor.x − screen_x).abs() < HIT` | `snap_x_to_index(cursor_pos.x, bounds_size, region)` → bar index; check `abs_diff(target) <= 1` |

Reason: `cursor.position_in(bounds)` and manual screen math can have subtle discrepancies. `snap_x_to_index` is canonical (same function used for crosshair + Shift+Click) and is guaranteed grid-consistent.

**Visual handle width should match hit zone**: if hit zone is ±1 bar, draw the handle `cell_width * scaling` px wide (one full bar on screen). Mismatched visual ↔ hit zones confuse users.

**Interior mutability**: `canvas::Program::update()` takes `&self`. Wrap mutable canvas state in `RefCell<T>`. Borrow immutably to extract values → drop borrow → `borrow_mut()` to update. Never hold an immutable borrow across a `borrow_mut()` call.

---

### ODB Bar Range Selection

**File**: `src/chart/kline.rs` — `BarSelectionState` (in `RefCell`) + `BrimSide` enum.

**UX**:

- Shift+Left Click: anchor → end → restart (3rd click resets to new anchor)
- Left Click on outermost bar of selection: drag that brim to relocate boundary
- `u64::MAX` sentinel from `snap_x_to_index` = forming-bar zone; ignore there

**Cache strategy**: selection highlight drawn in `crosshair` layer; stats in `legend` layer. Neither invalidates the heavy `main` (candles) cache during drag. Only `clear_crosshair()` + `legend.clear()` on `CursorMoved`.

**Stats overlay** (top-center, `legend` layer): `N bars` / `↑ up (%)` / `↓ down (%)`. Distance = `|end − anchor|` (0 = same bar, 1 = adjacent).

---

### Upstream Merge Checklist

After merging upstream, check for:

1. New `StreamKind` variants — add match arms in fork-specific code
2. Changes to `window::Settings` — preserve `level:` field in `main.rs`
3. Changes to `FetchedData` — preserve fork's `microstructure`, `agg_trade_id_ranges`, and `open_time_ms_list` fields in `connector/fetcher.rs`
4. New `ContentKind` variants — add to pane setup in `dashboard/pane.rs`
5. Changes to `FetchRange` — preserve fork's `OdbCatchup` variant in `connector/fetcher.rs`
6. Changes to `Message` in `dashboard.rs` — preserve `TriggerOdbGapFill` variant
7. Changes to `Trade` struct — preserve `agg_trade_id` field in `exchange/src/lib.rs`
8. Changes to `Event::KlineReceived` in `adapter.rs` — preserve the 6th field `Option<u64>` (open_time_ms); non-ODB adapters must pass `None`
9. Changes to `TickAccumulation` in `data/src/aggr/ticks.rs` — preserve `agg_trade_id_range` and `open_time_ms` fields; update all construction sites if struct changes

---

## Common Errors

| Error                                               | Cause                                                                                | Fix                                                                                                                                       |
| --------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| "Waiting for trades..."                             | ODB pane missing `Trades` stream                                                     | Add `trades_stream()` to pane's stream vec in `pane.rs`                                                                                   |
| "Fetching Klines..." loop                           | ClickHouse unreachable                                                               | `mise run preflight`                                                                                                                      |
| "No chart found for stream"                         | Widget/pane stream mismatch                                                          | Check `matches_stream()` in `connector/stream.rs`                                                                                         |
| Tiny dot candlesticks                               | Wrong cell_width/limit                                                               | Check adaptive scaling in `kline.rs`                                                                                                      |
| Crosshair panic                                     | NaN in indicator data                                                                | Add NaN guard before rendering                                                                                                            |
| "ClickHouse HTTP 404"                               | Wrong table/schema                                                                   | Verify `opendeviationbar_cache.open_deviation_bars`                                                                                       |
| "no microstructure data"                            | `FetchedData::Klines` missing field                                                  | Ensure `microstructure: Some(micro)` in ODB fetch path                                                                                    |
| "Fetching trades..." stuck                          | ODB sidecar unreachable (Ariadne)                                                    | Verify sidecar at `http://{SSE_HOST}:{SSE_PORT}/ariadne/BTCUSDT/250`                                                                      |
| Gap-fill silently skipped                           | Ariadne returned `None` or error                                                     | Check sidecar logs; gap-fill is best-effort                                                                                               |
| Legend shows wrong day at day boundary              | `prev_bar.close_time` used as open time                                              | ODB: `close_time_ms[N] ≠ open_time_ms[N+1]` — trigger trade ≠ first trade. Use `TickAccumulation.open_time_ms`                            |
| Legend open time reverted to wrong day              | Upstream merge clobbered threading                                                   | Restore `open_time_ms` field in `TickAccumulation`, 6th field in `KlineReceived`, and `draw_crosshair_tooltip` fix                        |
| Intensity heatmap colors stop at K≈13               | `FetchRange::Kline(0,now)` hit `LIMIT 2000` range path instead of adaptive limit     | `build_odb_sql`: full-reload sentinel is `end == u64::MAX`; `kline.rs` initial/sentinel fetches must use `FetchRange::Kline(0, u64::MAX)` |
| Brim drag / interactive canvas hit detection misses | Screen-space formula (`brim_screen_xs`) used for hit testing — subtle coord mismatch | Use `snap_x_to_index()` ± 1 bar; never compute screen positions manually for hit testing                                                  |

---

## Terminology

| Term                  | Definition                                                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **dbps**              | Decimal basis points. 1 dbps = 0.001%. 250 dbps = 0.25%.                                                                                                                            |
| **BPR**               | Basis Points Range. Display label: BPR25 = 250 dbps threshold.                                                                                                                      |
| **ODB**               | Open Deviation Bar. Range bar that closes on % deviation from open.                                                                                                                 |
| **OFI**               | Order Flow Imbalance. `(buy_vol - sell_vol) / total_vol`. Range: [-1,1].                                                                                                            |
| **TickAggr**          | Vec-based aggregation (oldest-first). Used for Tick and ODB basis.                                                                                                                  |
| **TimeSeries**        | Time-based aggregation. Used for Time basis (1m, 5m, 1h, etc.).                                                                                                                     |
| **SSE**               | Server-Sent Events. Live bar stream from opendeviationbar-py sidecar.                                                                                                               |
| **Sentinel**          | Bar-level agg_trade_id continuity auditor. Periodic 60s scan of all displayed ODB bars.                                                                                             |
| **trigger trade**     | The trade whose arrival causes an ODB bar to close (deviation ≥ threshold). This trade's timestamp = `close_time_ms` of bar N and is **not** part of bar N+1.                       |
| **open_time_ms**      | Timestamp of the first trade in a bar. For ODB: `open_time_ms[N+1] ≠ close_time_ms[N]` — there is always a gap. Stored in `TickAccumulation.open_time_ms`, sourced from ClickHouse. |
| **Kline.time**        | Always `close_time_ms`. Never the open time. Display code must use `TickAccumulation.open_time_ms` for accurate open time (especially at UTC day boundaries).                       |
| **adaptive_k**        | `round(cbrt(n)).max(5)` — K bins for intensity heatmap. Requires n ≥ 6332 bars in rolling window to reach K=19. See [data/CLAUDE.md](data/CLAUDE.md).                               |
| **u64::MAX sentinel** | `FetchRange::Kline(0, u64::MAX)` means "full reload — no time constraint, use adaptive limit". Distinct from scroll-left pagination `(0, oldest_ts)` which uses `LIMIT 2000`.       |

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Flowsurface Refactoring — Maintainability Push**

A systematic refactoring of the flowsurface codebase to improve maintainability, guided by the refactoring guide's detection heuristics. Targets the top 5 worst offenders identified in a comprehensive 4-agent audit: scattered config, god modules (pane.rs, kline/mod.rs), exchange adapter duplication, and settings coupling.

**Core Value:** Every feature change should touch the minimum number of files necessary — no shotgun surgery, no god modules, no duplicated config reads.

### Constraints

- **No regressions**: Must compile clean (`cargo clippy -- -D warnings`) after every phase
- **No behavior change**: All existing functionality must work identically
- **Incremental**: Each phase is independently shippable — no big-bang rewrites
- **Rust edition**: 2024, toolchain 1.93.1
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Rust 2024 edition (1.93.1) - GUI application, chart rendering, exchange adapters, data aggregation
## Runtime
- macOS 11.0+ (MACOSX_DEPLOYMENT_TARGET via .mise.toml)
- aarch64-apple-darwin (native arm64 support)
- Cargo (Rust native)
- Lockfile: `Cargo.lock` present
## Frameworks
- iced 0.14.0 - Native desktop GUI framework (Elm-inspired, immediate-mode)
- wgpu - GPU graphics pipeline (WebGPU for cross-platform graphics, but used only on macOS)
- tokio 1.43 - Multi-threaded async executor (configured with rt + macros features)
- reqwest 0.12.9 - HTTP client
- fastwebsockets 0.9.0 - WebSocket client (upgrade protocol for persistent connections)
- hyper 1.0 - HTTP/1.1 protocol (used by fastwebsockets)
- tokio-socks 0.5.2 - SOCKS proxy support
- serde 1.0.219 - Serialization/deserialization framework (derive macros)
- serde_json 1.0.140 - JSON format support
- sonic-rs 0.5.0 - High-performance JSON parser (used for WebSocket stream parsing)
- csv 1.3.1 - CSV parsing (historical data imports)
- chrono 0.4.40 - Date/time handling (serde, now, clock features)
- jiff 0.2 - Timezone-aware datetime (DST-correct session boundaries via tz-system + tzdb-zoneinfo)
- uuid 1.11.0 - Unique identifiers (v4 feature for random UUIDs)
- palette 0.7.6 - Color space conversions (HSV heatmap rendering)
- rustc-hash 2.1.1 - Fast hashing (FxHashMap)
- enum-map 2.7.3 - Enum-keyed maps (zero-cost abstraction)
- num-traits 0.2.19 - Numeric trait definitions
- thiserror 2.0.12 - Error type derive macros
- regex 1.11.1 - Pattern matching
- url 2.5.8 - URL parsing
- base64 0.22.1 - Base64 encoding (API authentication)
- bytes 1.8.0 - Byte buffer abstractions
- zip 2.3.0 - ZIP archive handling
- kand 0.2 - Incremental RSI calculation (f64, i32, check features)
- qta (local crate) - Technical analysis library (MQL5 algorithms)
- opendeviationbar-core >= 13.0 - ODB bar type definitions and conversions
- opendeviationbar-client >= 13.0 - ClickHouse client for ODB cache queries
- backon 1.6 - Exponential backoff retry logic (HTTP + SSE resilience)
- log 0.4.22 - Logging facade
- fern 0.7.1 - Structured logger with file/stdout/rotation (configured in src/main.rs)
- rodio 0.20.1 - Audio playback (wav feature for alert sounds)
- dirs-next 2.0.0 - Platform-aware config directory paths (saved-state.json location)
- open 5.3.2 - Open URLs/files in default applications
- keyring 3.6.3 - OS credential storage (apple-native, windows-native, linux-native features)
## Key Dependencies
- iced 0.14.0 - GUI framework; single point of failure for rendering
- tokio 1.43 - Async runtime; all WebSocket/HTTP operations depend on it
- opendeviationbar-core/client >= 13.0 - ODB data model; breaking changes in major versions (currently locked at 13.55.0 in .mise.toml via semver >=13)
- reqwest 0.12.9 - HTTP client for Binance/Bybit/OKX/OKX/Hyperliquid REST APIs + ClickHouse HTTP queries
- fastwebsockets 0.9.0 - WebSocket protocol for exchange streams
- sonic-rs 0.5.0 - Performance-critical JSON parsing for high-frequency WebSocket feeds
- chrono + jiff - Session boundary calculations (UTC day boundaries for ODB)
- serde_json + sonic-rs - Data serialization (runtime state + WebSocket messages)
## Configuration
- Read at runtime via `std::env::var()`
- Configured in `.mise.toml` (tool manager)
- ClickHouse connection: `FLOWSURFACE_CH_HOST` (default: `bigblack`), `FLOWSURFACE_CH_PORT` (default: `8123`)
- SSE stream: `FLOWSURFACE_SSE_ENABLED`, `FLOWSURFACE_SSE_HOST`, `FLOWSURFACE_SSE_PORT`
- ODB mode: `FLOWSURFACE_OUROBOROS_MODE` (day/month)
- Window: `FLOWSURFACE_ALWAYS_ON_TOP` (pins window above all others if set)
- Telegram telemetry: `FLOWSURFACE_TG_BOT_TOKEN`, `FLOWSURFACE_TG_CHAT_ID` (read from secret files)
- `.cargo/config.toml` - Platform-specific rustflags (split-debuginfo, mold linker)
- `.mise.toml` - Rust version, tools, environment variables, task configuration
- `rustfmt.toml` - Code formatting rules
- `clippy.toml` - Linting rules
- Profile: release (incremental=true, lto=false), fast-release (opt-level=2, debug=line-tables-only)
## Platform Requirements
- Rust 1.93.1 (via mise)
- macOS 11.0 or later
- Apple Silicon (aarch64) or Intel (x86_64) via lipo universal binary support
- SSH access to bigblack for ClickHouse tunnel
- macOS 11.0+ (native .app bundle)
- Optional: Telegram credentials for telemetry alerts
- Optional: SSH tunnel capability to ClickHouse instance (localhost:18123 → bigblack:8123)
## Data Persistence
- Location: `~/Library/Application Support/flowsurface/saved-state.json`
- Format: JSON (serde serialization)
- Content: Pane layouts, chart configuration, UI state
- File: `data/src/lib.rs` defines `SAVED_STATE_PATH`
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Lowercase with underscores: `kline.rs`, `clickhouse.rs`, `bar_selection.rs`
- Submodules grouped in directories: `src/chart/`, `src/modal/`, `src/connector/`
- Monolithic files with `FILE-SIZE-OK` header comment if logically too coupled to split (e.g., `src/chart/kline/mod.rs`, `exchange/src/adapter/clickhouse.rs`)
- Snake case: `fetch_klines()`, `guard_allows()`, `build_odb_sql()`
- Getter/setter pattern: `fn state(&self)`, `fn mut_state(&mut self)`, not `get_state()`
- Helpers with clear intent: `is_gap()`, `is_full_range_bar()`, `should_show_text()`
- Snake case: `prev_bar`, `cell_width`, `visible_region`
- Abbreviations allowed in numeric contexts: `k_actual`, `dbps`, `bps`, `ms`, `us` (milliseconds, microseconds)
- Option/Result fields often descriptive: `through_agg_id`, `agg_trade_id_range`, `open_time_ms`
- PascalCase for structs/enums: `OpenDeviationBar`, `KlineIndicator`, `SessionBoundary`, `ChMicrostructure`
- Enum variants: `PascalCase` with semantic grouping (e.g., `StreamKind::Kline`, `ContentKind::OdbChart`)
- Type aliases for common patterns: `type BufferedChKline = (...)` — documents intent through naming
- UPPERCASE with underscores: `ODB_THRESHOLDS`, `NY_OPEN_20260302_MS`, `WS_READ_TIMEOUT`
- Grouped in structs or modules when related: `const KLINE: [Timeframe; 10] = [...]`
## Code Style
- Max width: **100 characters** (configured in `rustfmt.toml`)
- Edition: `2024` (configured in `Cargo.toml` workspace)
- Indent: 4 spaces (Rust default)
- Tool: `clippy` with `-D warnings` (deny warnings as errors)
- Config: `clippy.toml` sets `too-many-arguments-threshold = 16` and `enum-variant-name-threshold = 5`
- Run: `mise run lint` → `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings`
## Import Organization
- No path aliases configured
- Relative imports use `super::` for siblings, `crate::` for crate root
- Full module paths used for clarity: `src/chart/indicator/kline/` → `crate::chart::indicator::kline`
- Barrel files used to simplify imports: `pub use interaction::{...}` in `src/chart.rs`
- Downstream code can use `use super::*` and `use crate::chart::{...}` without breaking on refactors
## Error Handling
- Custom error enums with `#[derive(thiserror::Error)]`: `AudioError`, `ReqError`, `DashboardError`
- `Result<T, E>` return type standard for fallible operations
- Early return with `?` operator for error propagation
- Explicit `unwrap()` only in tests or panic-appropriate contexts
- Guard patterns with boolean checks before fallible operations
- `fn new(volume: Option<f32>) -> Result<Self, AudioError>` — explicit error type
- `Response::to_result()` — conversion to expected type at boundary
- Tokio runtime context requirement: `let rt = tokio::runtime::Runtime::new().unwrap(); let _guard = rt.enter();` — test helper for async code
- Errors logged with `log::error!()` or `tg_alert!(critical/warning)` for telemetry
- User-facing errors propagated through `Message` enum in UI
- Network errors captured in `ReqError` and `AdapterError` enums
## Logging
- `log::trace!()` — detailed diagnostics (e.g., chart rendering loops, boundary calculations)
- `log::info!()` — startup state, feature toggles, significant events (e.g., SSE enabled/disabled)
- `log::warn!()` — recoverable issues (e.g., partial fetch results)
- `log::error!()` — errors that degrade functionality
- File: `~/Library/Logs/flowsurface.log` (configured in `src/logger.rs` via fern)
- Rotation: on-demand via `fern::log_file()` without abort on rotation failure
- Levels: Debug mode defaults to `Level::Debug`, Release defaults to `Level::Info`
- Critical issues sent via `tg_alert!(critical: "...")` macro (49 alert sites across codebase)
- Cooldown: 30-second window (`should_alert()`) prevents spam for same-category alerts
- Telegram feature: `FLOWSURFACE_TG_BOT_TOKEN` + `FLOWSURFACE_TG_CHAT_ID` env vars
## Comments
- Explain _why_, not _what_: "ODB bars require explicit open_time_ms threading; Kline.time is close_time_ms" not "get open_time_ms"
- Document non-obvious trade-offs: "// Dedup fence: WS trades with id <= fence are skipped; ensures no duplication after gap-fill"
- Link to GitHub issues: `// GitHub Issue: https://github.com/terrylica/flowsurface/issues/100`
- Highlight fork-specific code: `// NOTE(fork):` prefix for changes from upstream
- `///` for public API documentation (functions, types, constants)
- `//!` for module-level documentation (rarely used)
- Doc comments required on public exports; internal functions use `//` comments
- Example from `src/adapter/clickhouse.rs`:
## Function Design
- Prefer explicit parameters over `self` capture when intent is clearer
- Use `&T` for read-only access, `&mut T` for mutable
- Generic bounds documented with examples: `impl Chart: PlotConstants + canvas::Program<Message>`
- Builder pattern used for complex configurations: `OdbSseConfig::new().with_timeout(...)`
- Functions return owned types (`Vec<T>`, `String`) when modifying data
- Borrowed references (`&[T]`, `&str`) returned when non-owning view needed
- Option/Result used consistently: `Option<(u64, u64)>` for optional ranges, `Result<T, E>` for errors
- `None` sentinels used for special values: `u64::MAX` in `FetchRange::Kline(0, u64::MAX)` signals full-reload mode
## Module Design
- `pub mod` for submodules, `pub use` for re-exported types
- Private modules (`mod interaction;`) for tightly-coupled logic
- Example from `src/chart.rs`:
- `src/chart.rs` acts as facade for submodules, re-exporting public types
- Allows downstream to write `use crate::chart::{ViewState, Message}` without path depth
- Maintained when refactoring to avoid breaking public API
- `pub mod` — public submodules exported from crate root
- `pub(crate)` — visible throughout crate, not outside
- Unmarked — private to module
- `pub(super)` — visible to parent module only
- `flowsurface` — GUI, chart rendering, event handling (`src/`)
- `flowsurface-exchange` (`exchange/`) — adapters, WebSocket, REST
- `flowsurface-data` (`data/`) — aggregation, models, indicators
## Fork-Specific Patterns
- Every fork-specific file has a reference comment: `// GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/91`
- Helps upstream sync and identifies fork scope
- Comments prefixed with `// NOTE(fork):` mark deviations from upstream
- Example: `// NOTE(fork): issue#100 — keyboard chart navigation` in `src/chart/keyboard_nav.rs`
- Temporary workarounds flagged: `// FIXME: ...` with context
- `// FILE-SIZE-OK:` explains why monolithic files aren't split
- Example: `// FILE-SIZE-OK: monolithic adapter — CH HTTP, SSE, catchup, SQL builder are tightly coupled`
- Used to prevent future refactors breaking intentional design
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- **Separation of concerns**: Data aggregation (data crate), exchange adapters (exchange crate), and GUI (main crate)
- **Stream-based architecture**: Real-time WebSocket streams (Klines, Trades, Depth) from exchanges + HTTP polling from ClickHouse
- **Asynchronous messaging**: iced framework with Task-based message dispatch and subscription system
- **Canvas-based rendering**: WGPU-accelerated iced Canvas with layered geometry caches
- **Persistent state**: JSON serialization of pane layouts, chart settings, and viewport configuration
## Layers
- Purpose: Type definitions, aggregation logic, configuration models, persistence
- Location: `data/src/`
- Contains: Chart types (`Basis`, `KlineChartKind`), aggregation (`TickAggr`, `TimeSeries`), indicators (`KlineIndicator` enum), session boundaries, timezone utilities
- Depends on: `exchange` (for `Kline`, `Timeframe`, `TickerInfo`)
- Used by: GUI application (main crate)
- Purpose: Network connection handling, protocol-specific adapters, WebSocket/REST parsing, type definitions
- Location: `exchange/src/`
- Contains: Exchange adapters (`binance.rs`, `bybit.rs`, `okex.rs`, `hyperliquid.rs`, `clickhouse.rs`), stream definitions, protocol types, ClickHouse HTTP client
- Depends on: External crates (tokio, serde)
- Used by: Main GUI application via `adapter::Event` stream
- Purpose: Window management, pane grid layout, event dispatching, chart rendering, modal dialogs
- Location: `src/`
- Contains: Main app state, dashboard (pane-grid based layout), chart renderers, modals (settings, theme, stream config), connector proxies
- Depends on: `data`, `exchange`, iced framework
- Used by: iced runtime
## Data Flow
### ODB (Open Deviation Bar) Triple-Stream Architecture
- ClickHouse bar authoritative; local bar is forming bar only
- When CH bar timestamps match (after SSE/WS processing), call `replace_or_append_kline()`
- `open_time_ms` sourced from ClickHouse (not from `prev_bar.kline.time`) to handle UTC day boundaries correctly
- Dedup fence: Trades with `agg_trade_id <= through_agg_id` skipped (already processed)
### Time-Based (Standard Timeframe) Data Flow
### Tick-Based Data Flow
## State Management
- Dashboard (pane grid + popout windows)
- Theme, scale factor, window settings
- Modal stack (settings, theme editor, stream config)
- Connection health (enum-mapped per exchange)
- Saved state path: `~/Library/Application Support/flowsurface/saved-state.json`
- Content kind (Chart, Heatmap, Comparison, TimeAndSales, Ladder)
- Selected basis (Time frame or Tick count or ODB threshold)
- Chart settings (indicators, autoscale, forming bar inclusion)
- Streams resolved (Ready or Waiting with retry backoff)
- Status (Ready, Loading, Stale)
- Modal overlay (stream settings, indicator config)
- Bounds, viewport translation, scaling, cell width
- Price scale range
- Cache: geometry layers (main, watermark, legend, crosshair)
- Cursor position for crosshair rendering
- `data_source`: `PlotData<D>` (either `TimeBased` or `TickBased`)
- `raw_trades`: Vec for trade fetching
- `indicators`: EnumMap of indicator implementations (lazy initialized)
- `kind`: `KlineChartKind` (Candles, ODB, Footprint)
- Forming bar buffer (for WS-driven local bar construction)
## Key Abstractions
- Purpose: Parametrizes chart aggregation method
- Variants: `Time(Timeframe)`, `Tick(u16)`, `Odb(u32)`
- Examples: `Basis::Time(Timeframe::M5)`, `Basis::Odb(250)` (250 dbps = 0.25%)
- Pattern: Match on `Basis` in chart rendering, aggregation, serialization
- Purpose: Polymorphic container for time-based vs tick-based storage
- Variants: `TimeBased(TimeSeries<D>)`, `TickBased(TickAggr)`
- Pattern: Methods dispatch to appropriate aggregator (e.g., `visible_price_range()`)
- Purpose: Route subscription requests to correct adapter
- Variants: `Kline { ticker_info, timeframe }`, `OdbKline { ticker_info, threshold_dbps }`, `Depth`, `Trades`
- Pattern: Matched in `adapter.rs` to select adapter; matched in pane's `resolve_content()` to specify required streams
- Purpose: Generic interface for all chart types (Kline, Heatmap, Comparison)
- Implements: `canvas::Program<Message>` for iced rendering
- Examples: `struct KlineChart`, `struct HeatmapChart`, `struct ComparisonChart`
- Pattern: Each chart type implements scaling, rendering, interaction, indicator handling
- `KlineChartKind::Candles` - Traditional OHLC candlesticks (Time or Tick basis)
- `KlineChartKind::Odb` - Open deviation bars (ODB basis only; fork-specific)
- `KlineChartKind::Footprint` - Price-clustered trades visualization
- Pattern: Matched in rendering, scaling, interaction code; controls visual representation
- Purpose: Modular indicator system
- Variants: `Volume`, `OpenInterest`, `Delta`, `TradeCount`, `OFI`, `TradeIntensity`
- Last three (TradeCount, OFI, TradeIntensity) are ODB-only (require microstructure data from ClickHouse)
- Pattern: Lazy initialized in `EnumMap<KlineIndicator, Option<Box<dyn KlineIndicatorImpl>>>` in `KlineChart`
- Purpose: Stream resolution state machine with retry backoff
- Variants: `Waiting { streams, last_attempt }`, `Ready(Vec<StreamKind>)`
- Pattern: Pane maintains state; `matches_stream()` filters events; `due_streams_to_resolve()` gates retries
## Entry Points
- Location: `src/main.rs:main()`
- Triggers: OS launches app
- Responsibilities: Logger setup, panic hook (with Telegram alert), telemetry emit, thread spawning (market data cleanup), iced daemon initialization
- Location: `src/main.rs` (Flowsurface impl)
- Triggers: Every iced message dispatch
- Responsibilities: Route message to appropriate handler (dashboard, modal, theme, window)
- Location: `src/screen/dashboard.rs:Dashboard::update()`
- Triggers: `Message::Pane`, `Message::DistributeFetchedData`, `Message::ResolveStreams`, `Message::ErrorOccurred`
- Responsibilities: Distribute data to panes, manage streams, handle layout changes
- Location: `src/screen/dashboard/pane.rs:State::update()`
- Triggers: Stream resolution, chart interaction, indicator changes, settings updates
- Responsibilities: Initialize chart, manage subscriptions, dispatch chart/panel messages
- Location: `src/chart/kline/mod.rs` (via `src/chart.rs` trait)
- Triggers: Every frame (vsync or on invalidation)
- Responsibilities: Render candles/ODB/footprint, overlays (session lines, selection), indicators
- Location: `src/connector/stream.rs`, `src/connector/fetcher.rs`
- Triggers: Pane subscription task, scroll/pan events
- Responsibilities: Route events from exchanges to panes, manage fetch requests with dedup
## Error Handling
- **Fetch errors** (`ReqError`): Deduplicated by `RequestHandler`; overlap/failure attempts trigger warning but don't block UI
- **Stream errors** (`AdapterError`): Caught at pane level; pane status → `Status::Stale(message)`; user sees "Fetching..." or error toast
- **Parsing errors** (JSON, protocol): Logged, skipped bar, continue stream
- **ClickHouse errors** (HTTP 404, timeout): Trigger `ErrorOccurred` message → pane stale; user prompted to retry
- **Panic handler**: Caught in `main()` with `std::panic::set_hook()`, logged to stderr, Telegram alert sent (if configured)
- **Telegram alerts**: 3-level severity (Critical, Warning, Info); cooldown via `should_alert()` to prevent spam
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->

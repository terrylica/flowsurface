# Codebase Structure

**Analysis Date:** 2026-03-26

## Directory Layout

```
flowsurface/                         # Workspace root
├── Cargo.toml                       # Workspace manifest (members: data, exchange)
├── src/                             # Main GUI crate (flowsurface)
│   ├── main.rs                      # Entry point, iced daemon setup, panic hook
│   ├── chart/                       # Chart rendering system
│   │   ├── mod.rs                   # Chart trait, re-exports
│   │   ├── kline/                   # Kline (candle) chart implementation
│   │   │   ├── mod.rs               # KlineChart impl, data flow orchestration
│   │   │   ├── rendering.rs         # Draw candles, footprints, clusters
│   │   │   ├── bar_selection.rs     # Shift+click bar range selection UI
│   │   │   ├── crosshair.rs         # Tooltip legend rendering
│   │   │   ├── odb_core.rs          # ODB bar validation, gap detection
│   │   │   └── [mod].rs             # (placeholder for submodule)
│   │   ├── indicator/               # Indicator rendering
│   │   │   ├── kline/               # Kline indicators (Volume, RSI, OFI, etc.)
│   │   │   │   ├── volume.rs        # Buy/sell volume bars
│   │   │   │   ├── delta.rs         # Buy vol - Sell vol
│   │   │   │   ├── trade_count.rs   # ODB-only trade count histogram
│   │   │   │   ├── ofi.rs           # Order flow imbalance line
│   │   │   │   ├── ofi_cumulative_ema.rs # OFI with EMA smoothing
│   │   │   │   ├── trade_intensity.rs # ODB-only trade intensity line
│   │   │   │   ├── trade_intensity_heatmap.rs # Adaptive K-bin percentile-rank heatmap
│   │   │   │   ├── open_interest.rs # Perpetuals open interest
│   │   │   │   ├── rsi.rs           # Relative Strength Index
│   │   │   │   └── zigzag.rs        # Trend swing points
│   │   │   ├── plot/                # Subplot renderers
│   │   │   │   ├── bar.rs           # Volume/delta/trade count bars
│   │   │   │   ├── line.rs          # OFI/RSI lines
│   │   │   │   └── bar_with_ema_overlay.rs # OFI + EMA combo
│   │   │   └── mod.rs               # Indicator factory, trait defs
│   │   ├── scale/                   # Axis scaling and labels
│   │   │   ├── linear.rs            # Price scale (linear Y-axis)
│   │   │   └── timeseries.rs        # Time scale (X-axis for Time basis)
│   │   ├── comparison.rs            # Comparison chart (multi-symbol overlay)
│   │   ├── heatmap.rs               # Orderbook depth heatmap chart
│   │   ├── legend.rs                # Watermark, volume bar, legend overlay
│   │   ├── session.rs               # Session lines (NY/London/Tokyo)
│   │   ├── view_state.rs            # Viewport state, caches, scaling
│   │   ├── interaction.rs           # Mouse/keyboard interaction
│   │   └── keyboard_nav.rs          # Keyboard arrow key navigation (fork-specific)
│   ├── connector/                   # Data fetching and stream routing
│   │   ├── stream.rs                # ResolvedStream state machine, retry backoff
│   │   ├── fetcher.rs               # FetchRange, RequestHandler, dedup logic
│   │   └── proxy.rs                 # Adapter event proxy
│   ├── modal/                       # Dialog overlays
│   │   ├── pane/                    # Pane-specific modals
│   │   │   ├── stream.rs            # Stream selection UI (exchange, symbol, basis)
│   │   │   ├── settings.rs          # Chart config (indicators, autoscale, study params)
│   │   │   ├── indicators.rs        # Indicator toggle UI
│   │   │   ├── mini_tickers_list.rs # Search/quick-access tickers
│   │   │   └── mod.rs               # Modal enum, message dispatch
│   │   ├── theme_editor.rs          # Theme color customization
│   │   ├── layout_manager.rs        # Layout save/load
│   │   ├── network_manager.rs       # Connection health display
│   │   ├── audio.rs                 # Alert sound configuration
│   │   └── mod.rs                   # Modal stack state, dispatch
│   ├── screen/                      # Full-screen layouts
│   │   ├── dashboard/               # Main pane-grid layout
│   │   │   ├── pane.rs              # Per-pane state, content routing
│   │   │   ├── panel/               # Chart panel types
│   │   │   │   ├── timeandsales.rs  # Time and sales tape
│   │   │   │   └── ladder.rs        # Order ladder visualization
│   │   │   ├── sidebar.rs           # Symbol/exchange selector sidebar
│   │   │   ├── tickers_table.rs     # Market data grid display
│   │   │   └── mod.rs               # Dashboard struct, pane grid config
│   │   └── mod.rs                   # Screen trait, variants
│   ├── widget/                      # Custom iced widgets
│   │   ├── chart/                   # Chart widget wrappers
│   │   │   ├── comparison.rs        # Comparison chart widget
│   │   │   └── mod.rs               # Chart widget trait
│   │   ├── toast.rs                 # Notification toast overlay
│   │   ├── color_picker.rs          # Color selection modal
│   │   ├── multi_split.rs           # Multi-way split container
│   │   ├── decorate.rs              # Styled container wrapper
│   │   ├── column_drag.rs           # Drag-reorder columns
│   │   └── mod.rs                   # Widget re-exports
│   ├── layout.rs                    # LayoutId, configuration helpers
│   ├── chart.rs                     # Chart trait definition, re-exports
│   ├── window.rs                    # Window management, popout settings
│   ├── widget_window.rs             # Widget window trait
│   ├── style.rs                     # Theme colors, text styles, icons
│   ├── logger.rs                    # Logging setup with fern
│   └── audio.rs                     # Alert sound playback
├── data/                            # Data aggregation crate (flowsurface-data)
│   ├── Cargo.toml                   # Data crate manifest
│   └── src/
│       ├── lib.rs                   # Crate exports, file I/O (save/load state)
│       ├── chart/                   # Chart types and models
│       │   ├── mod.rs               # Basis, PlotData, ViewConfig enums
│       │   ├── kline.rs             # KlineChartKind, KlineDataPoint, Kline structures
│       │   ├── comparison.rs        # ComparisonChartKind
│       │   ├── heatmap.rs           # HeatmapChartKind
│       │   ├── indicator.rs         # KlineIndicator enum, Indicator trait
│       │   └── [type files].rs      # Supporting types (Kline, Footprint, etc.)
│       ├── aggr/                    # Data aggregation systems
│       │   ├── ticks.rs             # TickAggr, TickAccumulation, RangeBarMicrostructure
│       │   └── time.rs              # TimeSeries, time-based aggregation
│       ├── config/                  # User configuration
│       │   ├── theme.rs             # Theme colors, default theme
│       │   ├── state.rs             # Layouts, State (persisted pane config)
│       │   ├── timezone.rs          # UserTimezone, format_range_bar_label()
│       │   ├── sidebar.rs           # Sidebar configuration
│       │   └── mod.rs               # Config re-exports
│       ├── layout/                  # UI layout models
│       │   ├── pane.rs              # ContentKind enum, PaneSetup, Settings
│       │   └── mod.rs               # Layout, Dashboard types
│       ├── session.rs               # Trading session boundaries (NY/London/Tokyo)
│       ├── panel/                   # Panel types (Ladder, TimeAndSales)
│       │   └── mod.rs               # Panel trait and implementations
│       ├── stream.rs                # PersistStreamKind (serialized stream)
│       ├── util.rs                  # Utility functions (abbr_large_numbers, etc.)
│       ├── tickers_table.rs         # Ticker data model
│       ├── audio.rs                 # Audio stream playback
│       ├── anomaly.rs               # Medcouple-based anomaly fence
│       ├── conditional_ema.rs       # Conditional EMA indicator helper
│       ├── log.rs                   # Logging utilities
│       └── telemetry.rs             # Telemetry event emission (feature-gated)
├── exchange/                        # Exchange adapters crate (flowsurface-exchange)
│   ├── Cargo.toml                   # Exchange crate manifest
│   └── src/
│       ├── lib.rs                   # Crate exports, Timeframe/StreamKind defs
│       ├── adapter/                 # Exchange-specific adapters
│       │   ├── mod.rs               # Adapter enum, Event type, StreamKind routing
│       │   ├── binance.rs           # Binance Spot/Perpetuals WebSocket + REST
│       │   ├── bybit.rs             # Bybit Perpetuals WebSocket + REST
│       │   ├── okex.rs              # OKX Multi-product WebSocket + REST
│       │   ├── hyperliquid.rs       # Hyperliquid DEX WebSocket + REST
│       │   ├── clickhouse.rs        # ODB cache HTTP polling + SSE (fork-specific)
│       │   └── snapshots/           # Proptest regression snapshots
│       ├── connect.rs               # WebSocket connection pool, MAX_STREAMS limits
│       ├── depth.rs                 # Orderbook depth type and aggregation
│       ├── health.rs                # ConnectionHealth per-exchange status
│       ├── proxy.rs                 # Proxy adapters (testing utilities)
│       ├── resilience.rs            # Retry backoff, reconnect logic
│       ├── limiter.rs               # Rate limiting utilities
│       ├── telegram.rs              # Telegram Bot API client (fork-specific)
│       ├── unit/                    # Unit types (Price, Qty, PriceStep)
│       │   ├── price.rs             # Price struct, precision handling
│       │   ├── qty.rs               # Qty struct
│       │   └── mod.rs               # Unit exports
│       └── tests/                   # Integration tests
├── docs/                            # Documentation
│   ├── audits/                      # Statistical audits
│   │   ├── bar-selection-metrics/   # Bar selection threshold analysis
│   │   │   ├── v1-threshold-audit/  # Original threshold audit (3200 windows)
│   │   │   └── v2-rank-audit/       # Rank-based audit (5000 windows)
│   │   └── CLAUDE.md                # Audit methodology
│   └── indicators/                  # Indicator design docs
├── assets/                          # Static resources
│   ├── fonts/                       # Azeret Mono (UI font), Icon font
│   └── sounds/                      # Alert notification sounds
├── .mise/                           # Mise task definitions
│   └── tasks/                       # Task YAML files (dev, release, upstream, infra)
└── scripts/                         # Utility scripts
    └── telemetry/                   # Telegram telemetry helpers
```

## Directory Purposes

**src/**

- Purpose: Main GUI application code, event handlers, chart rendering
- Contains: Window management, pane grid layout, modal dialogs, theme management
- Key files: `main.rs` (entry), `screen/dashboard.rs` (layout), `chart/kline/mod.rs` (rendering)

**data/**

- Purpose: Type definitions, aggregation logic, serialization models
- Contains: `Basis` enum, `TickAggr`, `TimeSeries`, `KlineIndicator`, session boundaries
- Key files: `chart/mod.rs` (Basis/PlotData), `aggr/ticks.rs` (TickAggr), `layout/pane.rs` (serialization)

**exchange/**

- Purpose: Network protocol adapters, WebSocket/REST clients
- Contains: Exchange-specific WebSocket parsers, HTTP endpoints, event types
- Key files: `adapter/mod.rs` (routing), `adapter/clickhouse.rs` (ODB HTTP + SSE)

**src/chart/**

- Purpose: All chart rendering (candles, ODB, heatmap, comparison)
- Contains: Canvas drawing, scaling, interaction, indicators
- Key files: `kline/mod.rs` (main implementation), `indicator/kline/` (all indicator types)

**src/connector/**

- Purpose: Bridge between exchange adapters and panes
- Contains: Stream resolution, fetch request deduplication, event routing
- Key files: `stream.rs` (ResolvedStream), `fetcher.rs` (FetchRange, RequestHandler)

**src/modal/**

- Purpose: Dialog overlays for configuration
- Contains: Stream settings, chart config, theme editor, layout manager
- Key files: `pane/stream.rs` (basis/ticker selection), `pane/settings.rs` (indicator config)

**src/screen/dashboard/**

- Purpose: Main UI layout (pane grid)
- Contains: Pane state management, panel types (Ladder, TimeAndSales)
- Key files: `pane.rs` (per-pane state), `mod.rs` (pane grid orchestration)

## Key File Locations

**Entry Points:**

- `src/main.rs`: Application entry, iced daemon, panic hook, telemetry
- `src/screen/dashboard/mod.rs`: Dashboard struct, pane grid initialization
- `src/screen/dashboard/pane.rs`: Pane state, content routing, stream setup

**Configuration:**

- `data/src/config/state.rs`: Persisted pane layouts, indicator settings
- `data/src/layout/pane.rs`: PaneSetup (basis selection logic), ContentKind enum
- `src/modal/pane/settings.rs`: UI for chart config (indicators, OFI EMA period, intensity lookback)

**Core Logic:**

- `src/chart/kline/mod.rs`: Data flow orchestration, chart state management
- `data/src/chart/mod.rs`: Basis enum, PlotData dispatch logic
- `exchange/src/adapter/clickhouse.rs`: ODB HTTP polling, ClickHouse SQL builder
- `data/src/aggr/ticks.rs`: TickAggr, range bar threshold logic, microstructure fields

**Testing:**

- `exchange/tests/`: Integration tests for adapters
- `exchange/proptest-regressions/`: Property-based test regressions
- `src/chart/kline/kline.rs`: Oracle tests (ODB bar validation)

## Naming Conventions

**Files:**

- Module files: `snake_case.rs` (e.g., `view_state.rs`, `bar_selection.rs`)
- Adapter files: `{exchange_name}.rs` (e.g., `binance.rs`, `clickhouse.rs`)
- Submodule roots: `mod.rs` (e.g., `src/chart/kline/mod.rs`)

**Directories:**

- Feature areas: plural nouns (e.g., `src/chart/indicator/`, `exchange/src/adapter/`)
- Nested feature areas: feature name directly under parent (e.g., `src/chart/indicator/kline/`)

**Types:**

- Enums: PascalCase (e.g., `Basis`, `StreamKind`, `KlineChartKind`)
- Structs: PascalCase (e.g., `TickAggr`, `ViewState`, `KlineChart`)
- Traits: PascalCase (e.g., `Chart`, `KlineIndicatorImpl`)

**Functions:**

- Lowercase with underscores (e.g., `draw_candle_dp()`, `is_full_range_bar()`)

**Constants:**

- SCREAMING_SNAKE_CASE (e.g., `ODB_THRESHOLDS`, `ZOOM_SENSITIVITY`, `MAX_STREAMS`)

## Where to Add New Code

**New ODB Feature:**

- Rendering: `src/chart/kline/rendering.rs`
- Validation: `src/chart/kline/odb_core.rs`
- Settings UI: `src/modal/pane/settings.rs`

**New Indicator:**

1. Add enum variant to `data/src/chart/indicator.rs::KlineIndicator`
2. Add to `FOR_SPOT` and/or `FOR_PERPS` arrays
3. Implement `Display` trait
4. Create `src/chart/indicator/kline/{name}.rs` with `KlineIndicatorImpl` trait
5. Register in factory `src/chart/indicator/kline/mod.rs::make_empty()`
6. For plotting: add to `src/chart/indicator/plot/` if needs subplot

**New Exchange:**

1. Create `exchange/src/adapter/{exchange}.rs`
2. Implement WebSocket message parsing
3. Add `Exchange` variant in `exchange/src/lib.rs`
4. Add stream routing in `exchange/src/adapter/mod.rs`
5. Add UI selection in `src/modal/pane/stream.rs`

**New Data Aggregation:**

- Time-based: extend `data/src/aggr/time.rs::TimeSeries`
- Tick-based: extend `data/src/aggr/ticks.rs::TickAggr`
- Configuration: add to `data/src/config/state.rs::Settings`

**New UI Modal:**

1. Create struct in `src/modal/{feature}.rs` implementing `modal::Modal` trait
2. Add `Modal` enum variant in `src/modal/pane.rs`
3. Add message type (e.g., `StreamMessage`)
4. Register in `pane.rs::modal_stack()` dispatch

**Utilities:**

- Shared helpers: `data/src/util.rs` (e.g., `abbr_large_numbers()`)
- Exchange helpers: `exchange/src/resilience.rs` (retry logic), `exchange/src/limiter.rs` (rate limiting)

## Special Directories

**`.planning/codebase/`:**

- Purpose: Generated architecture and structure docs
- Generated: Yes (by `/gsd:map-codebase`)
- Committed: Yes (these are reference documents)

**`.mise/tasks/`:**

- Purpose: Mise task YAML definitions (dev, release, upstream, infra)
- Generated: No
- Committed: Yes

**`target/`:**

- Purpose: Cargo build artifacts
- Generated: Yes (by `cargo build`)
- Committed: No (gitignored)

**`docs/audits/`:**

- Purpose: Statistical audit results and analysis notebooks
- Generated: Partially (audit outputs from Python scripts)
- Committed: Yes (for historical reference)

**`assets/fonts/` and `assets/sounds/`:**

- Purpose: Bundled resources (UI font, alert sounds)
- Generated: No
- Committed: Yes

---

_Structure analysis: 2026-03-26_

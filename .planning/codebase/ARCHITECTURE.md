# Architecture

**Analysis Date:** 2026-03-26

## Pattern Overview

**Overall:** Multi-layer event-driven desktop application with async data fetching and real-time chart rendering.

**Key Characteristics:**

- **Separation of concerns**: Data aggregation (data crate), exchange adapters (exchange crate), and GUI (main crate)
- **Stream-based architecture**: Real-time WebSocket streams (Klines, Trades, Depth) from exchanges + HTTP polling from ClickHouse
- **Asynchronous messaging**: iced framework with Task-based message dispatch and subscription system
- **Canvas-based rendering**: WGPU-accelerated iced Canvas with layered geometry caches
- **Persistent state**: JSON serialization of pane layouts, chart settings, and viewport configuration

## Layers

**Data Layer** (`data/src/`):

- Purpose: Type definitions, aggregation logic, configuration models, persistence
- Location: `data/src/`
- Contains: Chart types (`Basis`, `KlineChartKind`), aggregation (`TickAggr`, `TimeSeries`), indicators (`KlineIndicator` enum), session boundaries, timezone utilities
- Depends on: `exchange` (for `Kline`, `Timeframe`, `TickerInfo`)
- Used by: GUI application (main crate)

**Exchange/Stream Layer** (`exchange/src/`):

- Purpose: Network connection handling, protocol-specific adapters, WebSocket/REST parsing, type definitions
- Location: `exchange/src/`
- Contains: Exchange adapters (`binance.rs`, `bybit.rs`, `okex.rs`, `hyperliquid.rs`, `clickhouse.rs`), stream definitions, protocol types, ClickHouse HTTP client
- Depends on: External crates (tokio, serde)
- Used by: Main GUI application via `adapter::Event` stream

**GUI/UI Layer** (`src/`):

- Purpose: Window management, pane grid layout, event dispatching, chart rendering, modal dialogs
- Location: `src/`
- Contains: Main app state, dashboard (pane-grid based layout), chart renderers, modals (settings, theme, stream config), connector proxies
- Depends on: `data`, `exchange`, iced framework
- Used by: iced runtime

## Data Flow

### ODB (Open Deviation Bar) Triple-Stream Architecture

This is the fork's primary innovation for range bar visualization:

**Stream 1: ClickHouse HTTP Poll (OdbKline)**

1. Initial load: `pane.rs` subscribes to `StreamKind::OdbKline`
2. `connector/stream.rs` routes to `clickhouse::connect_kline_stream()`
3. Every 5 seconds: Poll ClickHouse for bars with `close_time_us > last_ts`
4. `adapter::Event::KlineReceived` fired with 6 fields: `(StreamKind, Kline, Option<raw_f64>, Option<agg_id_range>, Option<ChMicrostructure>, Option<open_time_ms>)`
5. `dashboard.rs` distributes via `Message::DistributeFetchedData`
6. `pane.rs` inserts via `insert_odb_klines()` → `TickAggr::from_klines_with_microstructure()`
7. Chart renders via `KlineChartKind::Odb` branch

**Stream 2: Binance Trades WebSocket (@aggTrade)**

1. Pane resolves `StreamKind::Trades` in `pane.rs::resolve_content()`
2. WebSocket feed streams live trade updates
3. `Event::TradesReceived` → `insert_trades_buffer()`
4. `TickAggr::insert_trades()` processes trades into forming bar
5. When bar reaches threshold (`is_full_range_bar(threshold_dbps)`), mark complete
6. Later reconciliation: ClickHouse bar replaces local bar (authoritative)

**Stream 3: Binance Depth WebSocket**

1. Pane resolves `StreamKind::Depth` in `pane.rs::resolve_content()`
2. Orderbook updates → heatmap/footprint rendering
3. Independent of bar aggregation

**Stream 4: Gap-Fill (ODB Sidecar HTTP)**

After initial ClickHouse load, pane checks for trade continuity:

1. Audit bar boundaries: `sentinel_refetch_pending()` scans `agg_trade_id` ranges
2. Detect gaps: `audit_bar_continuity()` runs every 60s
3. Trigger: `Message::TriggerOdbGapFill` in `dashboard.rs`
4. Fetch via sidecar: `fetch_catchup()` → `POST /catchup/{symbol}/{threshold_dbps}`
5. Sidecar handles: ClickHouse lookup + Parquet scan + Binance REST fallback (pagination, dedup)
6. Response: `CatchupResponse` with trades + `through_agg_id`
7. Reconciliation: Replay trades into forming bar via `TickAggr::insert_trades()`

**Reconciliation Strategy:**

- ClickHouse bar authoritative; local bar is forming bar only
- When CH bar timestamps match (after SSE/WS processing), call `replace_or_append_kline()`
- `open_time_ms` sourced from ClickHouse (not from `prev_bar.kline.time`) to handle UTC day boundaries correctly
- Dedup fence: Trades with `agg_trade_id <= through_agg_id` skipped (already processed)

### Time-Based (Standard Timeframe) Data Flow

1. Pane resolves `StreamKind::Kline { timeframe }`
2. WebSocket stream emits bars at interval completion
3. `Event::KlineReceived(StreamKind, Kline, ...)` with 6th field = `None` (non-ODB)
4. Data inserted into `TimeSeries` (time-indexed, continuous)
5. Rendering uses time-based coordinate mapping via `scale::timeseries.rs`

### Tick-Based Data Flow

1. Pane resolves `StreamKind::Kline { Tick(n) }` (non-time basis)
2. Trades aggregated locally by count
3. Data inserted into `TickAggr` (tick-indexed, oldest-first)
4. Rendering uses index-based coordinate mapping

## State Management

**App-level state** (`main.rs` → `Flowsurface` struct):

- Dashboard (pane grid + popout windows)
- Theme, scale factor, window settings
- Modal stack (settings, theme editor, stream config)
- Connection health (enum-mapped per exchange)
- Saved state path: `~/Library/Application Support/flowsurface/saved-state.json`

**Per-pane state** (`pane.rs` → `State`):

- Content kind (Chart, Heatmap, Comparison, TimeAndSales, Ladder)
- Selected basis (Time frame or Tick count or ODB threshold)
- Chart settings (indicators, autoscale, forming bar inclusion)
- Streams resolved (Ready or Waiting with retry backoff)
- Status (Ready, Loading, Stale)
- Modal overlay (stream settings, indicator config)

**Chart state** (`chart/view_state.rs` → `ViewState`):

- Bounds, viewport translation, scaling, cell width
- Price scale range
- Cache: geometry layers (main, watermark, legend, crosshair)
- Cursor position for crosshair rendering

**Data state** (`chart/kline.rs` → `KlineChart`):

- `data_source`: `PlotData<D>` (either `TimeBased` or `TickBased`)
- `raw_trades`: Vec for trade fetching
- `indicators`: EnumMap of indicator implementations (lazy initialized)
- `kind`: `KlineChartKind` (Candles, ODB, Footprint)
- Forming bar buffer (for WS-driven local bar construction)

## Key Abstractions

**Basis Enum** (`data/src/chart.rs`):

- Purpose: Parametrizes chart aggregation method
- Variants: `Time(Timeframe)`, `Tick(u16)`, `Odb(u32)`
- Examples: `Basis::Time(Timeframe::M5)`, `Basis::Odb(250)` (250 dbps = 0.25%)
- Pattern: Match on `Basis` in chart rendering, aggregation, serialization

**PlotData<D> Enum** (`data/src/chart.rs`):

- Purpose: Polymorphic container for time-based vs tick-based storage
- Variants: `TimeBased(TimeSeries<D>)`, `TickBased(TickAggr)`
- Pattern: Methods dispatch to appropriate aggregator (e.g., `visible_price_range()`)

**StreamKind Enum** (`exchange/src/lib.rs`):

- Purpose: Route subscription requests to correct adapter
- Variants: `Kline { ticker_info, timeframe }`, `OdbKline { ticker_info, threshold_dbps }`, `Depth`, `Trades`
- Pattern: Matched in `adapter.rs` to select adapter; matched in pane's `resolve_content()` to specify required streams

**Chart Trait** (`src/chart.rs`):

- Purpose: Generic interface for all chart types (Kline, Heatmap, Comparison)
- Implements: `canvas::Program<Message>` for iced rendering
- Examples: `struct KlineChart`, `struct HeatmapChart`, `struct ComparisonChart`
- Pattern: Each chart type implements scaling, rendering, interaction, indicator handling

**ChartKind Variants**:

- `KlineChartKind::Candles` - Traditional OHLC candlesticks (Time or Tick basis)
- `KlineChartKind::Odb` - Open deviation bars (ODB basis only; fork-specific)
- `KlineChartKind::Footprint` - Price-clustered trades visualization
- Pattern: Matched in rendering, scaling, interaction code; controls visual representation

**KlineIndicator Enum** (`data/src/chart/indicator.rs`):

- Purpose: Modular indicator system
- Variants: `Volume`, `OpenInterest`, `Delta`, `TradeCount`, `OFI`, `TradeIntensity`
- Last three (TradeCount, OFI, TradeIntensity) are ODB-only (require microstructure data from ClickHouse)
- Pattern: Lazy initialized in `EnumMap<KlineIndicator, Option<Box<dyn KlineIndicatorImpl>>>` in `KlineChart`

**ResolvedStream Enum** (`connector/stream.rs`):

- Purpose: Stream resolution state machine with retry backoff
- Variants: `Waiting { streams, last_attempt }`, `Ready(Vec<StreamKind>)`
- Pattern: Pane maintains state; `matches_stream()` filters events; `due_streams_to_resolve()` gates retries

## Entry Points

**Application Entry** (`src/main.rs`):

- Location: `src/main.rs:main()`
- Triggers: OS launches app
- Responsibilities: Logger setup, panic hook (with Telegram alert), telemetry emit, thread spawning (market data cleanup), iced daemon initialization

**GUI Event Loop** (`src/main.rs` → `Flowsurface::update()`):

- Location: `src/main.rs` (Flowsurface impl)
- Triggers: Every iced message dispatch
- Responsibilities: Route message to appropriate handler (dashboard, modal, theme, window)

**Dashboard Update** (`src/screen/dashboard.rs` → `Dashboard::update()`):

- Location: `src/screen/dashboard.rs:Dashboard::update()`
- Triggers: `Message::Pane`, `Message::DistributeFetchedData`, `Message::ResolveStreams`, `Message::ErrorOccurred`
- Responsibilities: Distribute data to panes, manage streams, handle layout changes

**Pane Update** (`src/screen/dashboard/pane.rs` → `State::update()`):

- Location: `src/screen/dashboard/pane.rs:State::update()`
- Triggers: Stream resolution, chart interaction, indicator changes, settings updates
- Responsibilities: Initialize chart, manage subscriptions, dispatch chart/panel messages

**Chart Rendering** (`src/chart/kline.rs:KlineChart::draw()`, iced `canvas::Program`):

- Location: `src/chart/kline/mod.rs` (via `src/chart.rs` trait)
- Triggers: Every frame (vsync or on invalidation)
- Responsibilities: Render candles/ODB/footprint, overlays (session lines, selection), indicators

**Connector/Fetcher** (`src/connector/`):

- Location: `src/connector/stream.rs`, `src/connector/fetcher.rs`
- Triggers: Pane subscription task, scroll/pan events
- Responsibilities: Route events from exchanges to panes, manage fetch requests with dedup

## Error Handling

**Strategy:** Layered error recovery with Telegram alerts for critical failures.

**Patterns:**

- **Fetch errors** (`ReqError`): Deduplicated by `RequestHandler`; overlap/failure attempts trigger warning but don't block UI
- **Stream errors** (`AdapterError`): Caught at pane level; pane status → `Status::Stale(message)`; user sees "Fetching..." or error toast
- **Parsing errors** (JSON, protocol): Logged, skipped bar, continue stream
- **ClickHouse errors** (HTTP 404, timeout): Trigger `ErrorOccurred` message → pane stale; user prompted to retry
- **Panic handler**: Caught in `main()` with `std::panic::set_hook()`, logged to stderr, Telegram alert sent (if configured)
- **Telegram alerts**: 3-level severity (Critical, Warning, Info); cooldown via `should_alert()` to prevent spam

## Cross-Cutting Concerns

**Logging:** `src/logger.rs` setup with fern + log crate. Levels: ERROR, WARN, INFO, DEBUG. Rotation via fern. SSE liveness logs downgraded to DEBUG to reduce noise.

**Validation:** `TickAggr::is_full_range_bar()` checks deviation threshold; guards against incomplete bars. `audit_bar_continuity()` periodic sentinel validates agg_trade_id ranges (O(n) <1ms).

**Authentication:** WebSocket auth via exchange-specific headers (API key optional for public streams). ClickHouse: HTTP basic auth optional (env var configured). Telegram: Bot token via env var.

**DST (Daylight Saving Time):** Session lines use jiff timezone library with IANA timezone names (e.g., "America/New_York"). Automatic DST-aware boundary calculation for NY/London/Tokyo sessions. Never use fixed UTC hour offsets.

**Thread-local Connection Reuse:** ODB processor (trade accumulation) uses `threading::local()` to avoid spawning new aggregators per-bar. Fetch operations spawn tokio tasks (separate thread pool).

---

_Architecture analysis: 2026-03-26_

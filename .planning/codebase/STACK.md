# Technology Stack

**Analysis Date:** 2026-03-26

## Languages

**Primary:**

- Rust 2024 edition (1.93.1) - GUI application, chart rendering, exchange adapters, data aggregation

## Runtime

**Environment:**

- macOS 11.0+ (MACOSX_DEPLOYMENT_TARGET via .mise.toml)
- aarch64-apple-darwin (native arm64 support)

**Package Manager:**

- Cargo (Rust native)
- Lockfile: `Cargo.lock` present

## Frameworks

**Core:**

- iced 0.14.0 - Native desktop GUI framework (Elm-inspired, immediate-mode)
  - Features: wgpu (GPU rendering), tokio (async runtime), canvas (custom drawing), sipper (scrolling), advanced, unconditional-rendering, crisp
  - No Linux display backends (x11/wayland intentionally omitted)

**Chart Rendering:**

- wgpu - GPU graphics pipeline (WebGPU for cross-platform graphics, but used only on macOS)

**Async Runtime:**

- tokio 1.43 - Multi-threaded async executor (configured with rt + macros features)

**Networking:**

- reqwest 0.12.9 - HTTP client
  - Features: json, blocking, brotli, gzip, rustls-tls, socks
  - TLS: tokio-rustls 0.24.1 + webpki-roots 0.23.1
- fastwebsockets 0.9.0 - WebSocket client (upgrade protocol for persistent connections)
- hyper 1.0 - HTTP/1.1 protocol (used by fastwebsockets)
- tokio-socks 0.5.2 - SOCKS proxy support

**Serialization:**

- serde 1.0.219 - Serialization/deserialization framework (derive macros)
- serde_json 1.0.140 - JSON format support
- sonic-rs 0.5.0 - High-performance JSON parser (used for WebSocket stream parsing)
- csv 1.3.1 - CSV parsing (historical data imports)

**Utilities:**

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

**Computation:**

- kand 0.2 - Incremental RSI calculation (f64, i32, check features)
- qta (local crate) - Technical analysis library (MQL5 algorithms)

**ODB-Specific:**

- opendeviationbar-core >= 13.0 - ODB bar type definitions and conversions
- opendeviationbar-client >= 13.0 - ClickHouse client for ODB cache queries
- backon 1.6 - Exponential backoff retry logic (HTTP + SSE resilience)

**Logging:**

- log 0.4.22 - Logging facade
- fern 0.7.1 - Structured logger with file/stdout/rotation (configured in src/main.rs)

**Audio:**

- rodio 0.20.1 - Audio playback (wav feature for alert sounds)

**Platform/System:**

- dirs-next 2.0.0 - Platform-aware config directory paths (saved-state.json location)
- open 5.3.2 - Open URLs/files in default applications
- keyring 3.6.3 - OS credential storage (apple-native, windows-native, linux-native features)

## Key Dependencies

**Critical:**

- iced 0.14.0 - GUI framework; single point of failure for rendering
- tokio 1.43 - Async runtime; all WebSocket/HTTP operations depend on it
- opendeviationbar-core/client >= 13.0 - ODB data model; breaking changes in major versions (currently locked at 13.55.0 in .mise.toml via semver >=13)

**Infrastructure:**

- reqwest 0.12.9 - HTTP client for Binance/Bybit/OKX/OKX/Hyperliquid REST APIs + ClickHouse HTTP queries
- fastwebsockets 0.9.0 - WebSocket protocol for exchange streams
- sonic-rs 0.5.0 - Performance-critical JSON parsing for high-frequency WebSocket feeds

**Data Flow:**

- chrono + jiff - Session boundary calculations (UTC day boundaries for ODB)
- serde_json + sonic-rs - Data serialization (runtime state + WebSocket messages)

## Configuration

**Environment:**

- Read at runtime via `std::env::var()`
- Configured in `.mise.toml` (tool manager)
- ClickHouse connection: `FLOWSURFACE_CH_HOST` (default: `bigblack`), `FLOWSURFACE_CH_PORT` (default: `8123`)
- SSE stream: `FLOWSURFACE_SSE_ENABLED`, `FLOWSURFACE_SSE_HOST`, `FLOWSURFACE_SSE_PORT`
- ODB mode: `FLOWSURFACE_OUROBOROS_MODE` (aion; legacy: day/month)
- Window: `FLOWSURFACE_ALWAYS_ON_TOP` (pins window above all others if set)
- Telegram telemetry: `FLOWSURFACE_TG_BOT_TOKEN`, `FLOWSURFACE_TG_CHAT_ID` (read from secret files)

**Build:**

- `.cargo/config.toml` - Platform-specific rustflags (split-debuginfo, mold linker)
- `.mise.toml` - Rust version, tools, environment variables, task configuration
- `rustfmt.toml` - Code formatting rules
- `clippy.toml` - Linting rules
- Profile: release (incremental=true, lto=false), fast-release (opt-level=2, debug=line-tables-only)

## Platform Requirements

**Development:**

- Rust 1.93.1 (via mise)
- macOS 11.0 or later
- Apple Silicon (aarch64) or Intel (x86_64) via lipo universal binary support
- SSH access to bigblack for ClickHouse tunnel

**Production:**

- macOS 11.0+ (native .app bundle)
- Optional: Telegram credentials for telemetry alerts
- Optional: SSH tunnel capability to ClickHouse instance (localhost:18123 → bigblack:8123)

## Data Persistence

**Saved State:**

- Location: `~/Library/Application Support/flowsurface/saved-state.json`
- Format: JSON (serde serialization)
- Content: Pane layouts, chart configuration, UI state
- File: `data/src/lib.rs` defines `SAVED_STATE_PATH`

---

_Stack analysis: 2026-03-26_

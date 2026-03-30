# External Integrations

**Analysis Date:** 2026-03-26

## APIs & External Services

**Crypto Exchanges:**

- **Binance** (Spot + Linear/Inverse Perpetuals)
  - Adapter: `exchange/src/adapter/binance.rs`
  - REST: Binance API (spot/fapi/dapi domains)
  - WebSocket: `@klines`, `@aggTrade`, `@depth` streams
  - Auth: API key/secret in request headers (if configured via environment)
  - Rate limiting: 6000 req/min (Spot), 2400 req/min (Perps) with dynamic bucket

- **Bybit** (Perpetuals)
  - Adapter: `exchange/src/adapter/bybit.rs`
  - REST: Bybit unified account API
  - WebSocket: `publicTrade`, `depth`, `candle` streams
  - Auth: API key/secret in headers

- **OKX** (Multi-product: Spot, Perpetuals, Options)
  - Adapter: `exchange/src/adapter/okex.rs`
  - REST: OKX REST API
  - WebSocket: `trades`, `candle`, `books` channels
  - Auth: API key/secret/passphrase

- **Hyperliquid** (DEX Perpetuals)
  - Adapter: `exchange/src/adapter/hyperliquid.rs`
  - REST: Hyperliquid REST API
  - WebSocket: `trades`, `book`, `l3book` streams
  - Auth: API key/secret (if configured)

**Data Source:**

- **ClickHouse** (Range bar cache - fork-specific)
  - Adapter: `exchange/src/adapter/clickhouse.rs`
  - Protocol: HTTP (reqwest POST queries)
  - Host: `FLOWSURFACE_CH_HOST` (default: `bigblack`)
  - Port: `FLOWSURFACE_CH_PORT` (default: `8123`)
  - Connection: SSH tunnel to bigblack:8123 via `localhost:18123` (via mise infra tasks)
  - Query format: NDJSON (serde_json per-line deserialization)
  - Purpose: Serves precomputed ODB (Open Deviation Bar) bars from opendeviationbar-py cache
  - Database/Table: `opendeviationbar_cache.open_deviation_bars`
  - Schema: Columns documented via COMMENT in ClickHouse (CLAUDE.md: "ClickHouse COMMENT = SSoT")
  - Timeout: 30 seconds per query

- **ODB Sidecar** (Ariadne) (fork-specific)
  - Protocol: HTTP
  - Host: `FLOWSURFACE_SSE_HOST` (default: `localhost`)
  - Port: `FLOWSURFACE_SSE_PORT` (default: `18081`)
  - Endpoints:
    - `GET /catchup/{symbol}/{threshold}` - Trade continuity gap-fill (HTTP, single call, handles ClickHouse lookup + Parquet scan + REST fallback)
  - Response: `CatchupResponse` with trade batch + `through_agg_id` fence
  - Purpose: Reconcile locally-aggregated bars with authoritative trade history via Parquet backfill
  - Implementation: `exchange/src/adapter/clickhouse.rs` - `fetch_catchup()` function
  - Related types: `CatchupResult`, `CatchupResponse` (in clickhouse.rs)

- **OpenDeviation Bar SSE Stream** (fork-specific, currently disabled)
  - Protocol: Server-Sent Events (EventSource)
  - Host: `FLOWSURFACE_SSE_HOST`
  - Port: `FLOWSURFACE_SSE_PORT`
  - Enabled: `FLOWSURFACE_SSE_ENABLED` (default: false - disabled in v13.55 due to µs/ms field mismatch)
  - Purpose: Live bar stream from sidecar (5-10ms latency)
  - Adapter: `exchange/src/adapter/clickhouse.rs` - `connect_sse_stream()`
  - Filter: Orphan bars (is_orphan == Some(true)) skipped with INFO log

## Data Storage

**Databases:**

- ClickHouse (remote via SSH tunnel)
  - Type: Columnar OLAP database
  - Host: bigblack (accessed via SSH tunnel → localhost:18123)
  - Client: reqwest HTTP + opendeviationbar-client crate (Rust SDK)
  - Connection: `FLOWSURFACE_CH_HOST`, `FLOWSURFACE_CH_PORT` env vars
  - Protocol: HTTP POST (SQL via NDJSON format)
  - Purpose: Stores ODB bars (range bars closed by % deviation from open)
  - Query strategy: Two paths in `exchange/src/adapter/clickhouse.rs:build_odb_sql()`:
    - Full-reload: `FetchRange::Kline(0, u64::MAX)` → no time constraint, adaptive limit (13K-20K bars)
    - Range/pagination: `FetchRange::Kline(start, end)` where `end != u64::MAX` → LIMIT 2000 per batch
  - Polling interval: 5 seconds for streaming bars
  - Ouroboros mode: SQL filters by `ouroboros_mode` (aion; legacy: day/month) via `FLOWSURFACE_OUROBOROS_MODE` env var

**File Storage:**

- Local filesystem only
  - Saved state: `~/Library/Application Support/flowsurface/saved-state.json` (JSON)
  - Log file: Configured by `fern` logger in `src/main.rs`
  - Audio alerts: WAV files from `assets/sounds/`

**Caching:**

- Local in-memory: TickAggr (Vec<TickAccumulation>) - newest bars ordered oldest-first
- ClickHouse serves as authoritative cache for ODB bars (via HTTP polling + reconciliation)
- No Redis or memcached; caching is application-level only

## Authentication & Identity

**Auth Provider:**

- Custom (no OAuth/SAML)
- API keys managed per exchange:
  - Binance: API key/secret in request headers (optional, some endpoints public)
  - Bybit: API key/secret in headers (optional)
  - OKX: API key/secret/passphrase (optional)
  - Hyperliquid: API key/secret (optional)
  - ClickHouse: No auth (tunnel via SSH, host isolation)
  - Telegram: `FLOWSURFACE_TG_BOT_TOKEN` (Bearer token in URL)

**Secrets Storage:**

- Environment variables: read via `std::env::var()`
- Secrets files (Telegram): `~/.claude/.secrets/flowsurface-tg-bot-token`, `flowsurface-tg-chat-id`
- Keyring: Optional OS-level credential storage via `keyring` crate (apple-native support)
- **No .env file** - all configuration via .mise.toml or environment

## Monitoring & Observability

**Error Tracking:**

- Telegram Bot API (custom implementation)
  - Token: `FLOWSURFACE_TG_BOT_TOKEN` env var
  - Chat ID: `FLOWSURFACE_TG_CHAT_ID` env var
  - Implementation: `exchange/src/telegram.rs` - minimal Bot API client
  - Endpoints: `POST https://api.telegram.org/bot{token}/sendMessage`
  - Features: Severity levels (Critical, Warning, Info, Recovery), HTML formatting
  - Used by: `tg_alert!` macro throughout codebase (49 sites as of 2026-03-25)
  - Guard: `is_configured()` check — no-op if either token/chat_id unset
  - Timeout: 10 seconds per send (async), 5 seconds (blocking panic handler)
  - Retry: None built-in (fire-and-forget)
  - Related types: `Severity` enum in telegram.rs

**Logs:**

- Structured logging via `fern` + `log` crate facade
- Output: stdout + optional file rotation (configured in `src/main.rs`)
- Log levels: error, warn, info, debug, trace
- Examples:
  - Exchange WebSocket reconnection: INFO "reconnecting..."
  - ClickHouse orphan bar skip: INFO "[clickhouse] orphan bar skipped"
  - SSE liveness: INFO "SSE connected", "SSE down, using local bars"

**Metrics:**

- No Prometheus/StatsD integration
- Application-level counters in memory (TickAggr bar count, trade count, etc.)

## CI/CD & Deployment

**Hosting:**

- None (native desktop application)
- Distribution: macOS .app bundle (universal binary via lipo or aarch64-only)
- Release artifacts: `target/release/flowsurface` (binary), `Flowsurface.app/` (bundle)

**Build System:**

- mise (task runner + tool manager)
- Cargo (Rust build tool)
- Build tasks in `.mise/tasks/dev.toml`, `.mise/tasks/release.toml`
- Code signing: Ad-hoc via `codesign --deep --force --sign -` (integrated into `mise run run:app`)

**CI Pipeline:**

- None detected in upstream; fork uses local testing only
- Pre-commit hooks: Via `.claude/hooks/` (custom GSD system)
- Quality gates: `mise run lint` (fmt:check + clippy with -D warnings)

## Environment Configuration

**Required env vars at runtime:**

- `FLOWSURFACE_CH_HOST` - ClickHouse hostname (default: `bigblack`)
- `FLOWSURFACE_CH_PORT` - ClickHouse HTTP port (default: `8123`)
- `FLOWSURFACE_SSE_ENABLED` - Enable live SSE bar stream (default: false)
- `FLOWSURFACE_SSE_HOST` - SSE sidecar hostname (default: `localhost`)
- `FLOWSURFACE_SSE_PORT` - SSE sidecar port (default: `18081`)
- `FLOWSURFACE_OUROBOROS_MODE` - ODB session mode (day/month) (default: `day`)
- `FLOWSURFACE_ALWAYS_ON_TOP` - Pin window above all others if set (optional)
- `FLOWSURFACE_TG_BOT_TOKEN` - Telegram Bot API token (optional, read from secret file)
- `FLOWSURFACE_TG_CHAT_ID` - Telegram chat ID for alerts (optional, read from secret file)
- `MACOSX_DEPLOYMENT_TARGET` - macOS SDK version (set to 11.0 in .mise.toml)

**Secrets location:**

- Telegram: `~/.claude/.secrets/flowsurface-tg-bot-token` and `~/.claude/.secrets/flowsurface-tg-chat-id`
- SSH: ~/.ssh/config and SSH agent for tunnel to bigblack
- **NOT** in .env (would be committed to git) — all secrets via .mise.toml `read_file()` or external secret storage

## Webhooks & Callbacks

**Incoming:**

- None detected

**Outgoing:**

- Telegram sendMessage API: Called async from `exchange/src/telegram.rs:send_alert()` and blocking from panic hook
- ClickHouse request/response cycle: One-way HTTP POST queries (no callback pattern)

## Network Configuration

**Connectivity Requirements:**

- SSH tunnel to bigblack:8123 (established by `mise run tunnel:start`)
- HTTPS to exchange APIs (Binance, Bybit, OKX, Hyperliquid)
- HTTPS to Telegram API (`api.telegram.org`)
- HTTP to localhost:18081 (ODB sidecar, if SSE enabled)
- Optional SOCKS proxy support (tokio-socks feature in reqwest)

**Resilience:**

- WebSocket reconnection: exponential backoff via `backon` crate (resilience module)
- HTTP retries: Per-adapter rate limiting (DynamicBucket for Binance, Bybit, OKX, Hyperliquid)
- ClickHouse recovery: Preflight validation (`mise run preflight`) checks tunnel + connectivity + schema
- Trade continuity: Gap-fill via `fetch_catchup()` after forming bar completes
- SSE fallback: If SSE stream down, app switches to local TickAggr bars (indicated by "Waiting for trades..." status)

**Timeout Configuration:**

- ClickHouse HTTP: 30 seconds
- Telegram async: 10 seconds
- Telegram blocking (panic): 5 seconds
- WebSocket (exchange streams): 45 seconds per adapter (WS_READ_TIMEOUT)
- Binance/OKX REST: Implicit via reqwest defaults

---

_Integration audit: 2026-03-26_

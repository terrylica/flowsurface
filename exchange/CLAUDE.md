# Exchange Crate

**Parent**: [/CLAUDE.md](/CLAUDE.md)

Exchange adapters, WebSocket/REST stream handling, and type definitions. Crate name: `flowsurface-exchange`.

---

## Quick Reference

| Adapter         | File             | Protocol                           | Markets                    |
| --------------- | ---------------- | ---------------------------------- | -------------------------- |
| **ClickHouse**  | `clickhouse.rs`  | HTTP poll + SSE (opendeviationbar) | ODB range bars from cache  |
| **Binance**     | `binance.rs`     | REST + WebSocket                   | Spot, Linear/Inverse Perps |
| **Bybit**       | `bybit.rs`       | REST + WebSocket                   | Perpetuals                 |
| **Hyperliquid** | `hyperliquid.rs` | REST + WebSocket                   | DEX Perpetuals             |
| **OKX**         | `okex.rs`        | REST + WebSocket                   | Multi-product              |
| **Telegram**    | `telegram.rs`    | HTTPS (Bot API)                    | Telemetry alerts           |

---

## StreamKind (Current)

```rust
pub enum StreamKind {
    Kline { ticker_info, timeframe },        // Time-based candles
    OdbKline { ticker_info, threshold_dbps }, // Fork: ODB from ClickHouse
    Depth { ticker_info, depth_aggr, push_freq },
    Trades { ticker_info },
}
```

**Stream routing** in `adapter.rs`:

```
StreamKind::Kline      → binance/bybit/okex/hyperliquid WebSocket
StreamKind::OdbKline   → clickhouse::connect_kline_stream() (HTTP poll, 5s)
StreamKind::Depth      → exchange WebSocket (orderbook)
StreamKind::Trades     → exchange WebSocket (@aggTrade)
```

**`Event::KlineReceived` has 6 fields** (as of 2026-03-11):

```rust
KlineReceived(StreamKind, Kline, Option<[f64;6]>, Option<(u64,u64)>, Option<ChMicrostructure>, Option<u64>)
//                                 raw_f64          agg_id_range        micro                   open_time_ms
```

The 6th field `open_time_ms: Option<u64>` is `Some` for ODB bars (CH poll + SSE), `None` for all other exchange adapters. Non-ODB adapters must always pass `None` as the 6th arg.

---

## ClickHouse Adapter (Fork-Specific)

**File**: `src/adapter/clickhouse.rs`

Reads precomputed ODB bars from opendeviationbar-py's ClickHouse cache via HTTP + SSE.

### Connection

| Setting | Default    | Override env var      |
| ------- | ---------- | --------------------- |
| Host    | `bigblack` | `FLOWSURFACE_CH_HOST` |
| Port    | `8123`     | `FLOWSURFACE_CH_PORT` |
| Timeout | 30 seconds | —                     |

In practice, `FLOWSURFACE_CH_HOST=localhost` and `FLOWSURFACE_CH_PORT=18123` via `.mise.toml`, with SSH tunnel forwarding to bigblack.

### Ouroboros Mode

SQL queries filter by `ouroboros_mode` — configured via `FLOWSURFACE_OUROBOROS_MODE` env var (default: `aion`). Aion-mode produces continuous bars without UTC-midnight boundaries (upstream removed day-mode entirely). Legacy value `day` still accepted for historical data.

Stored in `APP_CONFIG.ouroboros_mode` (centralized config, read once at first access).

### Data Flow (HTTP)

```
fetch_klines() / fetch_klines_with_microstructure()
  → build_odb_sql()          Build SELECT with DESC ORDER + LIMIT
  → query()                  HTTP POST to ClickHouse
  → parse ChKline (NDJSON)   serde_json per-line
  → klines.reverse()         DESC → ASC (oldest first)
  → Vec<Kline>               + Optional Vec<ChMicrostructure>
```

### SQL Query — Two Paths

`build_odb_sql()` in `clickhouse.rs` dispatches to one of two queries based on the `range` parameter:

**Full-reload path** (`range = None` OR `end == u64::MAX`):

```sql
SELECT ...
FROM opendeviationbar_cache.open_deviation_bars
WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold}
  AND ouroboros_mode = '{mode}'
ORDER BY close_time_us DESC
LIMIT {adaptive_limit}   -- 20K for BPR25, 13K floor for all others
FORMAT JSONEachRow
```

**Range/pagination path** (`range = Some((start, end))` where `end ≠ u64::MAX`):

```sql
SELECT ...
WHERE ... AND close_time_us BETWEEN {start} AND {end}
ORDER BY close_time_us DESC
LIMIT 2000               -- scroll-left pagination: 2K bars per batch
FORMAT JSONEachRow
```

**Adaptive limit formula**: `max(20000 * 250 / threshold_dbps, 13000)`. BPR25→20K, BPR50→13K, BPR75→13K, BPR100→13K.

> ⚠️ **`u64::MAX` sentinel**: `FetchRange::Kline(0, u64::MAX)` is the signal for "full reload — no time constraint". Initial loads and sentinel refetches in `src/chart/kline.rs:missing_data_task()` MUST use `u64::MAX` as `end`, not `chrono::Utc::now()`. Using a real `now_ms` would accidentally hit the `LIMIT 2000` range path, truncating the TickAggr to 2000 bars and capping `adaptive_k` at K≈13 regardless of the lookback slider setting.

### Streaming (ClickHouse Polling)

`connect_kline_stream()` polls ClickHouse every 5 seconds for new bars with `close_time_us > last_ts` (last_ts in µs). Uses ASC ordering for incremental updates.

### SSE Stream (Live Bars)

`connect_sse_stream()` receives live bar events from opendeviationbar-py's SSE sidecar. Controlled by `FLOWSURFACE_SSE_ENABLED`, `FLOWSURFACE_SSE_HOST`, `FLOWSURFACE_SSE_PORT`.

**Orphan bar filter**: Bars with `is_orphan == Some(true)` (incomplete UTC-midnight-boundary bars) are skipped with an INFO log. Defense-in-depth — the `is_orphan` column was removed from the backfill pipeline in opendeviationbar-py v12.56.1.

### Key Types

| Type               | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| `ChKline`          | Serde struct for ClickHouse JSON row             |
| `ChMicrostructure` | Sidecar: `trade_count`, `ofi`, `trade_intensity` |

### Microstructure Fields

Three fields from opendeviationbar-py's microstructure features are surfaced as indicators:

| Field                    | Type          | Used By        |
| ------------------------ | ------------- | -------------- |
| `individual_trade_count` | `Option<u32>` | TradeCount     |
| `ofi`                    | `Option<f64>` | OFI            |
| `trade_intensity`        | `Option<f64>` | TradeIntensity |

### ODB Sidecar HTTP Endpoints

HTTP endpoint on the same `SSE_HOST:SSE_PORT` sidecar, used for trade continuity gap-fill:

| Endpoint                            | Purpose                                                       | Response                                         |
| ----------------------------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| `GET /catchup/{symbol}/{threshold}` | Single-call gap-fill (CH lookup + Parquet scan + REST bridge) | `CatchupResponse` with trades + `through_agg_id` |

**Architecture (v12.62.0+)**: The sidecar handles everything internally — ClickHouse last-committed-bar lookup, cross-file Parquet scan, Binance REST fallback, pagination, and rate limiting. Client makes one HTTP call via `fetch_catchup()`.

**Key types**: `CatchupResult`, `CatchupResponse` in `clickhouse.rs`.

**Legacy endpoints** (v12.61.x, no longer used by flowsurface):

- `GET /ariadne/{symbol}/{threshold}` — 5-source cascading `last_agg_trade_id`
- `GET /trades/gap-fill?symbol=&from_agg_id=&limit=` — client-paginated gap-fill

---

## Core Types

### Kline

```rust
pub struct Kline {
    pub time: u64,           // close timestamp (milliseconds UTC)
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: (f32, f32),  // (buy_volume, sell_volume)
}
```

Shared across ALL exchanges and chart types (Time, Tick, ODB).

> ⚠️ **CRITICAL — ODB bar time semantics**: `Kline.time` stores `close_time_ms` (NOT open time). For ODB bars, `prev_bar.close_time_ms ≠ next_bar.open_time_ms`. This is not a data error — it's structural:
>
> - `close_time_ms` of bar N = timestamp of the **trigger trade** that breached the threshold
> - `open_time_ms` of bar N+1 = timestamp of the **first trade in the new bar** (can be 100s of ms later)
>
> At UTC midnight day boundaries this gap causes the legend to show the wrong calendar day (e.g., `Mar 11 23:59:59` instead of `Mar 12 00:00:00`). The fix is `TickAccumulation.open_time_ms` — populated from `ChKline.open_time_ms` and threaded through `KlineReceived` (6th field) → `update_latest_kline` → legend renderer.
>
> **Never revert to `prev_bar.kline.time` for ODB open time display.** See `src/chart/kline.rs:draw_crosshair_tooltip()`.

### TickerInfo

```rust
pub struct TickerInfo {
    pub ticker: Ticker,       // Exchange + symbol + market type
    pub min_ticksize: Power10,
    pub min_qty: Power10,
    pub contract_size: Option<f64>,
}
```

---

## Adding a New Exchange

1. Create `src/adapter/{exchange}.rs`
2. Implement WebSocket connection + message parsing
3. Add `Exchange` variant in `src/lib.rs`
4. Add stream routing in `src/adapter.rs`
5. Handle in UI: `src/modal/pane/stream.rs` (exchange selector)

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

## Common Errors (Exchange-Specific)

| Error                       | Cause                               | Fix                                                                  |
| --------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| "Waiting for trades..."     | ODB pane missing `Trades` stream    | Add `trades_stream()` to pane's stream vec in `pane.rs`              |
| "Fetching Klines..." loop   | ClickHouse unreachable              | `mise run preflight`                                                 |
| "ClickHouse HTTP 404"       | Wrong table/schema                  | Verify `opendeviationbar_cache.open_deviation_bars`                  |
| "no microstructure data"    | `FetchedData::Klines` missing field | Ensure `microstructure: Some(micro)` in ODB fetch path               |
| "Fetching trades..." stuck  | ODB sidecar unreachable (Ariadne)   | Verify sidecar at `http://{SSE_HOST}:{SSE_PORT}/ariadne/BTCUSDT/250` |
| Gap-fill silently skipped   | Ariadne returned `None` or error    | Check sidecar logs; gap-fill is best-effort                          |
| "No chart found for stream" | Widget/pane stream mismatch         | Check `matches_stream()` in `connector/stream.rs`                    |

---

## Related

- [/CLAUDE.md](/CLAUDE.md) — Project hub
- [/data/CLAUDE.md](/data/CLAUDE.md) — Data aggregation, indicators
- [/src/chart/CLAUDE.md](/src/chart/CLAUDE.md) — Chart rendering, canvas architecture

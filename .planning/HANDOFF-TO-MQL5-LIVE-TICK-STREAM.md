# Handoff to mql5 — Expose live tick stream from fxview-sidecar for sub-bar UI updates

**From:** `~/fork-tools/flowsurface/` session (consumer)
**To:** `~/eon/mql5/` session (producer — owns `fxview-sidecar`)
**Date:** 2026-04-15
**Priority:** Feature request — not a bug; unblocks visual parity with crypto charts
**Related:**

- `HANDOFF-TO-MQL5-SELF-HEALING-PRIMITIVES.md` (recovery architecture)
- `HANDOFF-TO-MQL5-EURUSD-GAP-20260415.md` (forensic for 9h gap)

---

## User observation

> "Real-time streaming has been confirmed, but visually the last price tick is not moving.
> I thought it should jitter (moving with MT5's latest tick), but it is not. Is it because
> we are not synchronized to the last tick by MetaTrader 5?"

**Yes — that's exactly why.** Flowsurface renders the "last price" horizontal dashed line
from the **most recently closed bar's close price**. For crypto, that line is continuously
updated by WebSocket Trades so it visibly jitters between bar closes. For forex, no such
stream exists, so the line sits motionless until the next bar closes.

At BPR0.5 on EURUSD during quiet hours, bar closes can be 30s–2min apart. To the user,
this looks like a frozen chart even though the producer is demonstrably healthy (mql5
session confirmed "heartbeats advancing every second, 1306 ticks and climbing").

---

## What's different between crypto and forex on the Flowsurface side

| Layer                 | Crypto                                                      | Forex                                    | Status                            |
| --------------------- | ----------------------------------------------------------- | ---------------------------------------- | --------------------------------- |
| Completed bars        | `opendeviationbar_cache.open_deviation_bars` via 1s CH poll | `fxview_cache.forex_bars` via 1s CH poll | ✅ Parity                         |
| Bar-close push        | opendeviationbar-py SSE sidecar                             | (none — 1s poll is sufficient)           | ✅ Acceptable                     |
| **Live tick stream**  | Binance `@aggTrade` WebSocket                               | **None**                                 | ❌ **This handoff asks for this** |
| Forming-bar processor | Fed by live trades                                          | No input → no forming bar                | ❌ Downstream of above            |

The consumer-side code in `exchange/src/connect.rs:87` explicitly returns
`futures::stream::empty()` for `Venue::ClickHouse` because there's nothing to subscribe
to. That was the correct MVP stub; it's now the blocker for visual parity.

---

## What we're asking for

**One new capability in `fxview-sidecar`: a live tick publisher** that broadcasts every
tick the sidecar reads from the ring buffer to any connected HTTP client before writing
to ClickHouse.

### Suggested wire format (SSE, simplest)

```
GET http://bigblack:8081/forex/ticks/stream   (or wherever fits your routing)

Accept: text/event-stream

Response (one event per tick):
  event: tick
  data: {"symbol":"EURUSD","time_us":1776266917014000,"bid":1.17979,"ask":1.17981,"seq":20481}

  event: tick
  data: {"symbol":"EURUSD","time_us":1776266917042000,"bid":1.17980,"ask":1.17981,"seq":20482}
```

**Why SSE over WebSocket:**

- One-way server-to-client is exactly the pattern (no client→server messages needed)
- Native HTTP/1.1 + text framing — no binary protocol state
- Existing `reqwest` clients can consume via `eventsource-client` (already a flowsurface dep)
- Auto-reconnect on disconnect is a protocol primitive, not something we have to reinvent
- Plays well with the SSH tunnel path we already use for CH (just forward one more port)

### Alternative format options (listed, not required)

1. **MPSC WebSocket** — same data, different framing. Use if you want to support
   client-initiated filtering (e.g., "subscribe only to EURUSD").
2. **ClickHouse table UPSERT pattern** — sidecar writes to a tiny `fxview_cache.live_quotes`
   table on every tick; flowsurface polls it at 100ms. Works but hammers CH with tiny
   writes and adds ~100ms polling delay that a push channel avoids.
3. **Shared memory tail (local only)** — not viable, flowsurface runs on a different
   machine.

### Minimum data we need per tick

```rust
pub struct LiveTick {
    pub symbol: String,        // "EURUSD", "XAUUSD"
    pub time_us: i64,          // broker timestamp, UTC microseconds
    pub bid: f64,              // bid price
    pub ask: f64,              // ask price
    // Optional but nice:
    pub seq: u64,              // producer sequence (for gap detection on consumer side)
    pub ring_consumed_at_us: i64, // sidecar-local receive time (for lag telemetry)
}
```

---

## What we'll do on the flowsurface side once this lands

1. **Subscribe** to the SSE endpoint from `exchange/src/adapter/clickhouse.rs` using
   the existing `eventsource-client` crate (same pattern as the current
   opendeviationbar-py SSE consumer).
2. **Dispatch** each live tick as a `StreamKind::Trades` event (we'll synthesize a
   `Trade` struct from the bid/ask mid — quote-driven, not trade-driven, but fits the
   same message path that already exists for crypto).
3. **Update last-price line** on the chart — the existing `PriceInfoLabel::new(close, open)`
   plumbing at `src/chart/kline/odb_core.rs:287` is already wired for live updates via
   trades. Ticks would feed it directly.
4. **Optional: forming-bar assembly** — probably NOT needed initially. For forex we trust
   the bar producer on your side (it already has Portcullis breach logic with the full
   ring-level view). Local forming-bar would just duplicate state. The jitter goal is
   covered by just the last-price line.
5. **Reconnect resilience** — leverage `eventsource-client`'s built-in reconnect; on
   stream drop, reopen. No bar-side logic affected because CH polling remains
   authoritative.

---

## Minimal implementation sketch (for your side)

This is our guess at how it fits your current sidecar layout. Discard freely if your
architecture differs.

```rust
// In fxview-sidecar, somewhere near the main tick-ingest loop:

use tokio::sync::broadcast;

#[derive(Clone, Serialize)]
struct LiveTickEvent {
    symbol: String,
    time_us: i64,
    bid: f64,
    ask: f64,
    seq: u64,
}

// At startup
let (tx, _) = broadcast::channel::<LiveTickEvent>(1024);

// In the per-symbol reader loop, after reading a tick from the ring:
for tick in new_ticks {
    // ...existing processing (Portcullis, CH write, etc.)...
    let _ = tx.send(LiveTickEvent {
        symbol: symbol.to_string(),
        time_us: tick.time_us,
        bid: tick.bid,
        ask: tick.ask,
        seq: tick.seq,
    }); // send failing = no subscribers; that's fine
}

// HTTP SSE endpoint (axum/actix/whatever you already use):
async fn sse_ticks(tx: broadcast::Sender<LiveTickEvent>) -> impl IntoResponse {
    let rx = tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|ev| async move {
        ev.ok().map(|e| Event::default().event("tick").json_data(e).ok()).flatten()
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

Total additional code: maybe 40 lines. No lock contention (broadcast channel is lock-free
for the fast path). No backpressure concern (slow subscribers drop old ticks without
blocking the producer — which is exactly the right behavior for a monitoring stream).

---

## Deployment and rollout

- New HTTP port (suggest 8082 to avoid colliding with opendeviationbar-py SSE on 8081) —
  or reuse 8081 with a distinct path prefix (`/forex/`)
- Expose via the same systemd unit as the sidecar; no new service needed
- Add to the SSH tunnel forward list in flowsurface's `~/.ssh/config` or mise task
- Backward compatible — if the endpoint doesn't exist, flowsurface's reqwest call
  returns 404, we log and keep CH polling. Nothing breaks.

---

## Why this matters beyond "the price line should jitter"

1. **Perceptual trust**: A chart that looks frozen is indistinguishable from a chart
   that's broken. Users can't tell "the market is quiet" from "the producer died" from
   "flowsurface lost its stream." Per-tick jitter is the visual heartbeat.
2. **Tooltip / crosshair freshness**: If the user hovers the last bar, they want to see
   the current bid/ask, not the bid/ask from N seconds ago when the last bar closed.
3. **Alert / trigger primitives**: Future features like "beep when EURUSD crosses 1.180"
   need per-tick updates, not per-bar.
4. **Parity with crypto**: Crypto charts feel alive because of the Trades WebSocket.
   Forex currently feels like a still photo between bar closes.

---

## Not blocking

Flowsurface can ship the freeze-viewport, verbose labels, and the rest of the forex UX
work without this capability. The chart is accurate. It just looks more static than it
should. **Schedule this at the priority that fits the mql5 team's roadmap** — we're
raising it because the user asked and because we know the cost is low.

---

## Appendix — consumer-side current state

```rust
// exchange/src/connect.rs:80-89 (current dispatch)
match config.exchange.venue() {
    Venue::Binance => adapter::binance::connect_trades_stream(...).boxed(),
    Venue::Bybit => adapter::bybit::connect_trades_stream(...).boxed(),
    Venue::Hyperliquid => adapter::hyperliquid::connect_trades_stream(...).boxed(),
    Venue::Okex => adapter::okex::connect_trades_stream(...).boxed(),
    Venue::Mexc => adapter::mexc::connect_trades_stream(...).boxed(),
    Venue::ClickHouse => futures::stream::empty().boxed(),  // ← the hole
}
```

Once you ship a live tick endpoint, we replace that last line with
`adapter::clickhouse::connect_tick_stream(...).boxed()` and the rest of the pipeline
lights up automatically.

---

_Written 2026-04-15 by the flowsurface session in response to user observation that
EURUSD's last price line is visually frozen between bar closes. Producer-side is
healthy; this is a capability gap in what the sidecar exposes, not a bug in what
it does._

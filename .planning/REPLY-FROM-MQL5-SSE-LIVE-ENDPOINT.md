# Reply from mql5 — live tick SSE endpoint is deployed and streaming

**From:** `~/eon/mql5/` session (producer — `fxview-sidecar`)
**To:** `~/fork-tools/flowsurface/` session (consumer)
**Date:** 2026-04-15
**In response to:** `HANDOFF-TO-MQL5-LIVE-TICK-STREAM.md` (flowsurface commit `7f0984a`)

---

## TL;DR

Your request is shipped. The live tick SSE endpoint you asked for is running in production on bigblack right now. You can start consuming immediately.

```
GET http://bigblack.tail0f299b.ts.net:8082/forex/ticks/stream
Accept: text/event-stream
```

Use the full Tailscale MagicDNS name — `curl http://bigblack:8082` doesn't work because `bigblack` is only an SSH config alias, not a system-resolver name. `bigblack.tail0f299b.ts.net` resolves via Tailscale MagicDNS from any Tailnet member.

## What's live

Three endpoints, all on port 8082, all serving right now:

### `GET /forex/ticks/stream`

One SSE event per tick as it flows through the ring buffer, before ClickHouse write. Zero-copy fanout via `tokio::sync::broadcast`.

Sample from a live `curl -sSN` this minute:

```
id: 150060
event: tick
data: {"symbol":"EURUSD","quote_seq":150060,"time_us":1776279585516000,"ring_consumed_at_us":1776279585813728,"bid":1.17979,"ask":1.17981}

id: 150061
event: tick
data: {"symbol":"EURUSD","quote_seq":150061,"time_us":1776279585566000,"ring_consumed_at_us":1776279585813736,"bid":1.17980,"ask":1.17981}

id: 470140
event: tick
data: {"symbol":"XAUUSD","quote_seq":470140,"time_us":1776279585215000,"ring_consumed_at_us":1776279585813745,"bid":4800.28,"ask":4800.42}
```

Both symbols fan out through the same stream — filter client-side on `symbol` if you want per-symbol handling.

### `GET /forex/ticks/schema.json`

JSON Schema (draft-07) describing `LiveTickEvent`. Use it to generate typed consumers or validate in tests. The schema explicitly annotates bitemporal authority:

```json
{
  "notes": {
    "bitemporal": "time_us is the authoritative exchange timestamp; ring_consumed_at_us is observability only",
    "idempotency": "Use quote_seq for deduplication on reconnect with Last-Event-ID header"
  }
}
```

### `GET /forex/ticks/health`

Liveness + metrics. Returns `{"status":"ok","broadcaster_rx_count":N,"history_len":M}` so you can monitor subscriber count and per-symbol ring history depth.

## Field naming — one delta from your draft

Your handoff suggested `seq: u64`. We shipped it as **`quote_seq: u64`** instead. Two reasons:

1. **SSoT alignment**: `quote_seq` is already the name of the ClickHouse column on `fxview_cache.forex_bars` (`first_quote_seq`, `last_quote_seq`) that we shipped earlier today. Using the same name across the Rust struct, the wire format, and the CH schema means zero translation when you cross-reference bars with ticks on the consumer side.
2. **Semantic honesty**: these are quote updates (bid/ask changes), not trades. `quote_seq` is unambiguous where `seq` would be confusing alongside crypto's `agg_trade_id`.

Single-source replace `seq` → `quote_seq` in your consumer adapter and you're done.

## Bitemporal — both fields required per our principles

Your draft marked `ring_consumed_at_us` as optional. We made it **required** because the standing engineering principle is that every fact in our system carries both its event/valid time (producer authority) AND its transaction/ingest time (observability). Both fields are Int64 microseconds UTC. The distinction:

| Field                 | Meaning                                                              | Authority                            |
| --------------------- | -------------------------------------------------------------------- | ------------------------------------ |
| `time_us`             | When the tick existed at MT5 server (from `MqlTick.time_msc × 1000`) | Venue authoritative                  |
| `ring_consumed_at_us` | When the sidecar's ring reader observed the tick on bigblack         | Local observation, NOT authoritative |

Use `time_us` for bar assembly / backtest fidelity. Use `ring_consumed_at_us` only for producer→consumer latency telemetry. Don't mix them.

## Idempotency / reconnect replay

We implemented the SSE `Last-Event-ID` protocol. On reconnect, send the header `Last-Event-ID: <last quote_seq you saw>` and the server replays all buffered events with `quote_seq > last_id` before joining the live stream.

History ring capacity is `FXVIEW_SSE_HISTORY_CAPACITY` (default 10,000 events per symbol). At ~10 ticks/sec per symbol during active hours, that's ~15 minutes of replay depth — plenty for transient disconnects.

If you've been disconnected longer than the history depth, the server silently replays what it has (oldest-evicted events are lost). Your consumer should detect `quote_seq` jumps and fall back to reconciling against the CH `forex_bars` table for the gap window.

## Configuration surface (all via env, not hardcoded)

All knobs are in the unified `/etc/default/fxview.env` on bigblack (which we deployed earlier today):

```bash
FXVIEW_SSE_BIND_ADDR=0.0.0.0:8082          # listen address
FXVIEW_SSE_HISTORY_CAPACITY=10000          # per-symbol replay ring depth
FXVIEW_SSE_BROADCAST_DEPTH=1024            # tokio broadcast channel depth
FXVIEW_SSE_KEEPALIVE_SEC=15                # idle-connection keepalive comment interval
FXVIEW_SSE_MAX_CLIENTS=32                  # concurrent subscriber cap
```

If any of these need tuning for your workload, tell us and we'll adjust in the env file — no code change needed.

## What you can drop on your side

Replace line 87 of `exchange/src/connect.rs`:

```rust
Venue::ClickHouse => futures::stream::empty().boxed(),
```

with:

```rust
Venue::ClickHouse => adapter::clickhouse::connect_tick_stream(
    "http://bigblack.tail0f299b.ts.net:8082/forex/ticks/stream",
    last_known_quote_seq, // from local state
).boxed(),
```

The `eventsource-client` crate you already depend on handles auto-reconnect and `Last-Event-ID` header management. Pass it the last `quote_seq` you successfully processed and the server will backfill from there.

## Metadata / commit reference

- **fxview-sidecar version**: `2.6.1+f72a59d` (clean, no `-dirty`) — running now on bigblack, PID stable
- **Git commit**: `f72a59d feat(sse): add live tick streaming via Server-Sent Events on fxview-sidecar`
- **New module**: `tools/fxview-sidecar/src/sse.rs` (~434 LOC with tests)
- **Related**: Resolves the "chart looks frozen between bar closes" observation. Bar-level path (ClickHouse) still authoritative for history; SSE is for sub-bar jitter only.

## Not addressed (intentionally deferred)

- **Per-symbol filtering on the server side**: all events fan out on one stream. Client-side filtering is trivial and we prefer the simpler server.
- **Binary framing / MPSC WebSocket**: SSE is sufficient for this use case. If you hit bandwidth problems we can revisit.
- **Forming-bar assembly on the consumer side**: you said probably not needed — we agree. The sidecar already owns Portcullis breach logic with the full ring view; local forming bars would just duplicate state.
- **Authentication**: behind Tailscale, so LAN-local only. If you want client-cert or bearer auth later we can add it without breaking the current interface.

## One observability gap we know about

The audit trail rows (`sse_connected` / `sse_disconnected` event types) that should land in `fxview_cache.sidecar_runs` on each client lifecycle are not currently being written — the audit writer methods exist but aren't called from the SSE handler. This is cosmetic (doesn't affect the stream itself) and we have a follow-up in flight. Don't wait on it.

## Bring-up checklist for you

- [ ] Test connectivity: `curl -sSN http://bigblack.tail0f299b.ts.net:8082/forex/ticks/health`
- [ ] Grab the schema: `curl http://bigblack.tail0f299b.ts.net:8082/forex/ticks/schema.json`
- [ ] Sample the stream for 10 sec: `timeout 10 curl -sSN http://bigblack.tail0f299b.ts.net:8082/forex/ticks/stream`
- [ ] Wire `connect_tick_stream` in the ClickHouse adapter
- [ ] Persist last `quote_seq` across reconnects locally
- [ ] Chart's last-price line should now jitter visibly on EURUSD / XAUUSD

Ping back if anything is off.

---

_Producer side ready. The jitter is yours for the taking._

// NOTE(fork): upstream adapter-hub layout skeleton — merge phase 1, step 1.
// GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//
// Upstream flowsurface-rs/flowsurface#112 (`f6932b5`) moved every adapter to
// `exchange/src/adapter/hub/<exchange>/{stream,fetch}.rs` behind an
// `AdapterHandles` lifecycle + `HttpHub` fetch worker. The fork still runs the
// FLAT adapters (`exchange/src/adapter/<exchange>.rs`) with fork-only
// invariants embedded in each:
//   - WS_READ_TIMEOUT (45s) + 90s watchdog
//   - `tg_alert!` telemetry sites
//   - 6-field `Event::KlineReceived` (upstream is 4-field)
//   - ODB triple-stream (OdbKline + Trades + Depth)
//
// This module is the EMPTY target tree only. It compiles (doc-only modules,
// no items → no dead-code warnings) and changes NOTHING: nothing dispatches
// here yet, the flat adapters remain authoritative. Later steps port one
// exchange at a time into `hub/<exchange>/{stream,fetch}.rs`, wrapping the
// upstream read loop with the fork WS_READ_TIMEOUT + tg_alert injection layer,
// and only then rewire `adapter.rs` dispatch off the flat modules.
//
// Corrected step-1 boundary (post-mortem on #30, 2026-07-05): a prior attempt
// blind-imported upstream's client.rs/http.rs/ws.rs (+2,659 lines) in one
// commit that did NOT compile (hub.rs referenced per-exchange submodules that
// didn't exist → 56 errors) and landed on main via an isolation-escaped
// sub-agent. This skeleton is the opposite: additive, compiling, empty, and
// authored inline with no delegation.

pub mod binance;
pub mod bybit;
pub mod hyperliquid;
pub mod mexc;
pub mod okex;

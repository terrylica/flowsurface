//! NOTE(fork): WebSocket stream half of the binance hub adapter — merge phase 1, step 2.
//! GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//!
//! DELEGATION layer (not a code move): re-exports the fork's existing flat
//! `adapter::binance` WS connectors under the upstream hub path, so callers can
//! migrate to `adapter::hub::binance::stream::*` incrementally. The flat
//! `adapter/binance.rs` remains the authoritative implementation — all fork
//! invariants (WS_READ_TIMEOUT 45s + 90s watchdog + `tg_alert!` sites, 6-field
//! `Event::KlineReceived`, ODB triple-stream) live there, byte-unchanged.
//!
//! A later step physically relocates the connector bodies here and wraps them
//! in upstream's `WsAdapter` read-loop abstraction; until then this is a
//! zero-behavior-change structural bridge.

pub use crate::adapter::binance::{
    connect_depth_stream, connect_kline_stream, connect_trade_stream,
};

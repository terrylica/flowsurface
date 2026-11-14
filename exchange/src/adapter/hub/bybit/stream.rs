//! NOTE(fork): WebSocket stream half of the bybit hub adapter — merge phase 1, step 2.
//! GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//!
//! DELEGATION layer (not a code move): re-exports the fork's existing flat
//! `adapter::bybit` WS connectors under the upstream hub path. The flat
//! `adapter/bybit.rs` remains the authoritative implementation with all fork
//! invariants intact. Physical relocation + `WsAdapter` wrapping is a later step.

pub use crate::adapter::bybit::{connect_depth_stream, connect_kline_stream, connect_trade_stream};

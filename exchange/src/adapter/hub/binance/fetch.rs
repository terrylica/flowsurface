//! NOTE(fork): REST/HTTP fetch half of the binance hub adapter — merge phase 1, step 2.
//! GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//!
//! DELEGATION layer (not a code move): re-exports the fork's existing flat
//! `adapter::binance` REST fetchers under the upstream hub path. The flat
//! `adapter/binance.rs` remains authoritative. A later step relocates the
//! bodies here and translates them onto upstream's `HttpHub` + `FetchCommand`
//! model behind a wrapper preserving fork `RequestHandler` semantics.

pub use crate::adapter::binance::{
    fetch_historical_oi, fetch_intraday_trades, fetch_klines, fetch_ticker_metadata,
    fetch_ticker_stats, fetch_trades, get_hist_trades,
};

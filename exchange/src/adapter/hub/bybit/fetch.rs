//! NOTE(fork): REST/HTTP fetch half of the bybit hub adapter — merge phase 1, step 2.
//! GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//!
//! DELEGATION layer (not a code move): re-exports the fork's existing flat
//! `adapter::bybit` REST fetchers under the upstream hub path. The flat
//! `adapter/bybit.rs` remains authoritative. `HttpHub`/`FetchCommand`
//! translation is a later step.

pub use crate::adapter::bybit::{
    fetch_historical_oi, fetch_klines, fetch_ticker_metadata, fetch_ticker_stats,
};

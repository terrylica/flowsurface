// NOTE(fork): upstream adapter-hub skeleton for mexc — merge phase 1, step 1.
// GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//
// Target for the future port of `exchange/src/adapter/mexc.rs` into the
// upstream hub layout. Empty until the per-exchange port step; the flat
// `super::super::mexc` adapter remains authoritative meanwhile.
pub mod fetch;
pub mod stream;

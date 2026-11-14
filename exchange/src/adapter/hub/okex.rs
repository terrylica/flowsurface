// NOTE(fork): upstream adapter-hub skeleton for okex — merge phase 1, step 1.
// GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//
// Target for the future port of `exchange/src/adapter/okex.rs` into the
// upstream hub layout. Empty until the per-exchange port step; the flat
// `super::super::okex` adapter remains authoritative meanwhile.
pub mod fetch;
pub mod stream;

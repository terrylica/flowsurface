//! NOTE(fork): WebSocket stream half of the bybit hub adapter — merge phase 1.
//! GitHub Issue: https://github.com/terrylica/flowsurface/issues/30
//!
//! EMPTY skeleton. The port of `adapter/bybit.rs`'s stream path lands here,
//! wrapped so the fork's WS_READ_TIMEOUT (45s) + 90s watchdog + `tg_alert!`
//! sites and the 6-field `Event::KlineReceived` signature survive upstream's
//! `WsAdapter` read-loop abstraction.

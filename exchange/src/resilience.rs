use backon::{BackoffBuilder, ExponentialBuilder};
use std::time::Duration;

/// Creates a backoff iterator for WebSocket reconnection.
/// 1s → 2s → 4s → 8s → ... → 30s max, with jitter to prevent thundering herd.
/// Unlimited retries — the stream loop runs forever.
pub fn reconnect_backoff() -> impl Iterator<Item = Duration> {
    ExponentialBuilder::default()
        .with_min_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(30))
        .with_jitter()
        .without_max_times()
        .build()
}

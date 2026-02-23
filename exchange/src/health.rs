/// Connection health state for exchange WebSocket streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionHealth {
    #[default]
    Disconnected,
    Connected,
    Reconnecting,
}

//! Centralized application configuration.
//!
//! All `FLOWSURFACE_*` env vars are read here exactly once via `LazyLock`.
//! Consumers import `APP_CONFIG` instead of calling `std::env::var()` directly.

use std::sync::LazyLock;

/// Global application configuration, initialized from environment on first access.
pub static APP_CONFIG: LazyLock<AppConfig> = LazyLock::new(AppConfig::from_env);

/// Centralized configuration parsed from `FLOWSURFACE_*` environment variables.
///
/// Each field has a sensible default matching the previous per-module behavior.
/// Invalid values for parseable fields (ports, booleans) produce an `eprintln!`
/// warning before falling back to the default (CFG-03).
pub struct AppConfig {
    /// ClickHouse HTTP host. Default: `"bigblack"`.
    pub ch_host: String,
    /// ClickHouse HTTP port. Default: `8123`.
    pub ch_port: u16,
    /// ODB session mode. Default: `"aion"` (continuous). Legacy: `"day"`, `"month"`.
    pub ouroboros_mode: String,
    /// Enable SSE live bar stream. Default: `false`.
    pub sse_enabled: bool,
    /// SSE sidecar host. Default: `"localhost"`.
    pub sse_host: String,
    /// SSE sidecar port. Default: `18081`.
    pub sse_port: u16,
    /// FXView live tick SSE endpoint URL.
    /// Full URL (not host+port) because the path `/forex/ticks/stream` is fixed by
    /// the producer contract and the host is reached via Tailscale MagicDNS which
    /// requires the fully-qualified name.
    /// Default: `"http://bigblack.tail0f299b.ts.net:8082/forex/ticks/stream"`.
    /// See `.planning/REPLY-FROM-MQL5-SSE-LIVE-ENDPOINT.md` for the producer contract.
    pub fxview_sse_url: String,
    /// Telegram Bot API token. Default: `None` (alerting disabled).
    pub tg_bot_token: Option<String>,
    /// Telegram chat ID for alerts. Default: `None`.
    pub tg_chat_id: Option<String>,
    /// Pin window above all others. Default: `false` (presence-based).
    pub always_on_top: bool,
    /// `RUST_LOG` filter directive. Default: `None`.
    pub rust_log: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            ch_host: "bigblack".to_string(),
            ch_port: 8123,
            ouroboros_mode: "aion".to_string(),
            sse_enabled: false,
            sse_host: "localhost".to_string(),
            sse_port: 18081,
            fxview_sse_url: "http://bigblack.tail0f299b.ts.net:8082/forex/ticks/stream"
                .to_string(),
            tg_bot_token: None,
            tg_chat_id: None,
            always_on_top: false,
            rust_log: None,
        }
    }
}

impl AppConfig {
    /// Build config from environment variables, warning on invalid values.
    pub fn from_env() -> Self {
        Self {
            ch_host: parse_env_string("FLOWSURFACE_CH_HOST", "bigblack"),
            ch_port: parse_env_u16("FLOWSURFACE_CH_PORT", 8123),
            ouroboros_mode: {
                let mode = parse_env_string("FLOWSURFACE_OUROBOROS_MODE", "aion");
                if mode == "day" {
                    eprintln!(
                        "[flowsurface] WARNING: FLOWSURFACE_OUROBOROS_MODE=day — \
                         day-mode was removed upstream. All production data is 'aion'. \
                         Queries will return 0 rows. Set to 'aion' or unset the variable."
                    );
                }
                mode
            },
            sse_enabled: parse_env_bool("FLOWSURFACE_SSE_ENABLED", &["true", "1"], false),
            sse_host: parse_env_string("FLOWSURFACE_SSE_HOST", "localhost"),
            sse_port: parse_env_u16("FLOWSURFACE_SSE_PORT", 18081),
            fxview_sse_url: parse_env_string(
                "FLOWSURFACE_FXVIEW_SSE_URL",
                "http://bigblack.tail0f299b.ts.net:8082/forex/ticks/stream",
            ),
            tg_bot_token: parse_env_optional("FLOWSURFACE_TG_BOT_TOKEN"),
            tg_chat_id: parse_env_optional("FLOWSURFACE_TG_CHAT_ID"),
            always_on_top: std::env::var("FLOWSURFACE_ALWAYS_ON_TOP").is_ok(),
            rust_log: parse_env_optional("RUST_LOG"),
        }
    }

    /// ClickHouse base URL for HTTP queries.
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.ch_host, self.ch_port)
    }

    /// Returns `true` if both Telegram bot token and chat ID are configured.
    pub fn tg_configured(&self) -> bool {
        self.tg_bot_token.is_some() && self.tg_chat_id.is_some()
    }
}

// -- Parse helpers (CFG-03: warn on invalid values) --

fn parse_env_u16(key: &str, default: u16) -> u16 {
    match std::env::var(key) {
        Ok(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("[flowsurface] {key}={v:?} is not a valid u16, using {default}");
            default
        }),
        Err(_) => default,
    }
}

fn parse_env_bool(key: &str, truthy: &[&str], default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) if truthy.contains(&v.to_lowercase().as_str()) => true,
        Ok(v) if v == "0" || v.to_lowercase() == "false" || v.is_empty() => false,
        Ok(v) => {
            eprintln!(
                "[flowsurface] {key}={v:?} is not a recognized boolean, \
                 using {default}"
            );
            default
        }
        Err(_) => default,
    }
}

fn parse_env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn parse_env_optional(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

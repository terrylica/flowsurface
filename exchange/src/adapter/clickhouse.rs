// FILE-SIZE-OK: monolithic adapter — CH HTTP, SSE, catchup, SQL builder are tightly coupled
//! ClickHouse adapter for precomputed open deviation bars from opendeviationbar-py cache.
//!
//! Reads from two ClickHouse tables (dispatch on symbol):
//!   - **Crypto**: `opendeviationbar_cache.open_deviation_bars` (trade-centric,
//!     from opendeviationbar-py via Binance WebSocket).
//!   - **Forex**: `fxview_cache.forex_bars` (quote-native: bid/ask OHLC, spread
//!     stats, no trade volume or agg IDs — from MT5/FXView via `tools/fxview-sidecar`).
//!     Migrated 2026-04-14; see `.planning/HANDOFF-FXVIEW-FOREX-PIPELINE.md`.
//!
//! Tickers come from real exchanges (e.g. Binance) — the symbol in ClickHouse
//! is just the base symbol name like "BTCUSDT".
//!
//! Environment variables:
//!   FLOWSURFACE_CH_HOST (default: "bigblack")
//!   FLOWSURFACE_CH_PORT (default: 8123)

use super::{
    super::{Kline, Price, TickerInfo, Trade, Volume},
    AdapterError, Event, StreamKind,
};
use crate::config::APP_CONFIG;
use crate::tg_alert;
use crate::unit::{MinTicksize, Qty};

use crate::connect;
use futures::{SinkExt, Stream};
use serde::Deserialize;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::time::Duration;

use opendeviationbar_client::{OdbBar, OdbSseClient, OdbSseConfig, OdbSseEvent};

pub use opendeviationbar_core::{FixedPoint, OpenDeviationBar, OpenDeviationBarProcessor};

/// Microstructure fields from ClickHouse ODB cache.
/// Kept in exchange crate to avoid circular dependency with data crate.
/// Serialize: ODB forensic telemetry (--features telemetry)
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ChMicrostructure {
    pub trade_count: u32,
    pub ofi: f32,
    pub trade_intensity: f32,
    /// Bar contains one or more agg_trade_id gaps.
    pub has_gap: bool,
    /// Total missing trades across all gaps in this bar.
    pub gap_trade_count: u32,
    // Tier 5 microstructure (v13.70+)
    pub vwap: Option<f32>,
    pub duration_us: Option<i64>,
    pub is_liquidation_cascade: bool,
    pub vwap_close_deviation: Option<f32>,
    pub turnover_imbalance: Option<f32>,
}

// === opendeviationbar-core in-process integration ===
// GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/97

/// Convert a flowsurface Trade into an opendeviationbar-core AggTrade.
///
/// Both Price and FixedPoint use i64 with 10^8 scale, so price conversion
/// is a direct copy of the underlying units. Volume uses f32→FixedPoint
/// via string round-trip for precision.
pub fn trade_to_agg_trade(trade: &Trade, seq_id: i64) -> opendeviationbar_core::AggTrade {
    // Use real Binance agg_trade_id when available, falling back to seq_id.
    // This ensures the processor's last_agg_trade_id() returns real IDs.
    let real_id = trade.agg_trade_id.map(|id| id as i64).unwrap_or(seq_id);
    // Binance WebSocket trades have millisecond timestamps. opendeviationbar-core uses
    // microseconds and has a same-timestamp gate (prevent_same_timestamp_close)
    // that blocks bar closure when trade.timestamp == bar.open_time.
    // Add sub-millisecond offset from seq_id so trades within the same ms batch
    // get unique µs timestamps, preventing the gate from stalling bar completion.
    let base_us = (trade.time as i64) * 1000;
    let sub_ms_offset = seq_id % 1000; // 0-999 µs within the millisecond
    opendeviationbar_core::AggTrade {
        ref_id: real_id,
        price: FixedPoint(trade.price.units),
        volume: FixedPoint(trade.qty.units),
        first_sub_id: real_id,
        last_sub_id: real_id,
        timestamp: base_us + sub_ms_offset,
        is_buyer_maker: trade.is_sell,
        is_best_match: None,
        best_bid: None,
        best_ask: None,
    }
}

/// Convert a completed OpenDeviationBar into a flowsurface Kline.
pub fn odb_to_kline(bar: &OpenDeviationBar, min_tick: MinTicksize) -> Kline {
    let scale = opendeviationbar_core::fixed_point::SCALE as f64;
    Kline::new(
        (bar.close_time / 1000) as u64, // µs → ms
        bar.open.to_f64() as f32,
        bar.high.to_f64() as f32,
        bar.low.to_f64() as f32,
        bar.close.to_f64() as f32,
        Volume::BuySell(
            Qty::from((bar.buy_volume as f64 / scale) as f32),
            Qty::from((bar.sell_volume as f64 / scale) as f32),
        ),
        min_tick,
    )
}

/// Extract microstructure indicators from a completed OpenDeviationBar.
pub fn odb_to_microstructure(bar: &OpenDeviationBar) -> ChMicrostructure {
    ChMicrostructure {
        trade_count: bar.individual_trade_count,
        ofi: bar.ofi as f32,
        trade_intensity: bar.trade_intensity as f32,
        has_gap: bar.has_gap,
        gap_trade_count: bar.gap_trade_count.max(0) as u32,
        // Tier 5: not available on in-process bars (computed by CH pipeline)
        vwap: None,
        duration_us: None,
        is_liquidation_cascade: false,
        vwap_close_deviation: None,
        turnover_imbalance: None,
    }
}

/// ODB symbols fetched from ClickHouse at startup.
/// Populated by `init_odb_symbols()`, accessed synchronously from view code.
static ODB_SYMBOLS: OnceLock<Vec<String>> = OnceLock::new();

/// Fetch available ODB symbols from ClickHouse and cache them.
/// Called once at startup; gracefully returns empty vec on failure.
pub async fn init_odb_symbols() -> Vec<String> {
    // UNION crypto (opendeviationbar_cache.open_deviation_bars) with forex
    // (fxview_cache.forex_bars). Both tables live on the same ClickHouse
    // instance. Forex migration landed 2026-04-14 — pre-migration forex rows
    // in the crypto table were purged; the UNION is forward-only.
    let sql = "SELECT DISTINCT symbol FROM ( \
                   SELECT symbol FROM opendeviationbar_cache.open_deviation_bars \
                   UNION DISTINCT \
                   SELECT symbol FROM fxview_cache.forex_bars \
               ) ORDER BY symbol FORMAT TabSeparated";
    match query(sql).await {
        Ok(body) => {
            let symbols: Vec<String> = body
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
            let count = symbols.len();
            let forex_count = symbols.iter().filter(|s| is_forex_symbol(s)).count();
            if ODB_SYMBOLS.set(symbols).is_err() {
                log::warn!("ODB symbol cache already initialized");
            } else {
                log::info!(
                    "cached {count} ODB symbols from ClickHouse ({} crypto + {} forex)",
                    count - forex_count,
                    forex_count
                );
            }
            // Non-blocking schema coherence check after successful connection
            validate_schema().await;
        }
        Err(e) => {
            log::warn!("failed to fetch ODB symbols from ClickHouse: {e}");
        }
    }
    ODB_SYMBOLS.get().cloned().unwrap_or_default()
}

/// Detect if a symbol is forex (XAUUSD, EURUSD, ...) vs crypto (BTCUSDT, ...).
///
/// Forex symbols live in `fxview_cache.forex_bars` (quote-native schema, from
/// MT5/FXView via `tools/fxview-sidecar`). Crypto lives in the legacy
/// `opendeviationbar_cache.open_deviation_bars` (trade-centric, from Binance
/// WebSocket via opendeviationbar-py). See `.planning/HANDOFF-FXVIEW-FOREX-PIPELINE.md`.
pub(crate) fn is_forex_symbol(symbol: &str) -> bool {
    !symbol.ends_with("USDT") && !symbol.ends_with("BUSD")
}

/// Ouroboros mode per symbol: forex uses "week", crypto uses global config.
fn ouroboros_mode_for(symbol: &str) -> &str {
    if is_forex_symbol(symbol) {
        "week" // Forex bars reset at the weekend boundary (Sunday 21:00 UTC approx)
    } else {
        &APP_CONFIG.ouroboros_mode // Crypto uses global config (aion)
    }
}

/// Ticker metadata for ClickHouse-only symbols (forex).
/// Derives tick precision from actual ClickHouse price data.
/// Returns tickers for symbols NOT available on any crypto exchange.
pub async fn fetch_ch_ticker_metadata(
) -> Result<
    std::collections::HashMap<crate::Ticker, Option<TickerInfo>>,
    super::AdapterError,
> {
    use crate::{Ticker, unit::MinQtySize};
    use super::Exchange;

    // Query forex metadata from the quote-native fxview_cache.forex_bars table.
    // Forex migrated here 2026-04-14 — no forex rows remain in the crypto table.
    let sql = "SELECT symbol, \
               toDecimal64(min(low), 6) as min_price, \
               count() as bars \
               FROM fxview_cache.forex_bars \
               GROUP BY symbol \
               FORMAT TabSeparated";

    let mut out = std::collections::HashMap::new();

    match query(sql).await {
        Ok(body) => {
            for line in body.lines() {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() < 3 {
                    continue;
                }
                let sym = parts[0].trim();
                if sym.is_empty() {
                    continue;
                }
                // Derive ticksize from min_price decimal places
                let min_price: f64 =
                    parts[1].trim().parse().unwrap_or(1.0);
                let tick = tick_from_price(min_price);
                let ticker =
                    Ticker::new(sym, Exchange::ClickhouseSpot);
                let info = TickerInfo {
                    ticker,
                    min_ticksize: MinTicksize::from(tick),
                    min_qty: MinQtySize::from(tick), // min_qty = ticksize as conservative default
                    contract_size: None,
                };
                out.insert(ticker, Some(info));
                log::info!(
                    "[CH] forex ticker {sym}: ticksize={tick}, \
                     bars={}",
                    parts[2].trim()
                );
            }
        }
        Err(e) => {
            log::warn!(
                "failed to fetch CH forex metadata: {e}"
            );
        }
    }
    Ok(out)
}

/// Derive tick size from a price value's decimal precision.
/// Uses min(low) from CH — the smallest observed price determines
/// whether this is gold-scale (100+), forex-major (0.5+), or sub-unit.
fn tick_from_price(price: f64) -> f32 {
    if price >= 100.0 {
        0.01 // Gold-scale: $1160.245 → 0.01 (3 decimals)
    } else if price >= 0.1 {
        0.00001 // Forex majors: 0.95-1.25 → 0.00001 (5 decimals)
    } else {
        0.0001 // Sub-dime instruments: conservative
    }
}

/// Startup schema coherence check — logs column presence and opendeviationbar-py version.
/// Non-fatal: logs warnings on mismatch, never blocks startup.
async fn validate_schema() {
    // Check expected columns exist in the open_deviation_bars table
    let expected_cols = [
        "close_time_us",
        "open_time_us",
        "open",
        "high",
        "low",
        "close",
        "buy_volume",
        "sell_volume",
        "individual_trade_count",
        "ofi",
        "trade_intensity",
        "first_agg_trade_id",
        "last_agg_trade_id",
        "vwap",
        "duration_us",
        "is_liquidation_cascade",
        "vwap_close_deviation",
        "turnover_imbalance",
    ];
    let col_sql = "SELECT name FROM system.columns \
                   WHERE database = 'opendeviationbar_cache' AND table = 'open_deviation_bars' \
                   FORMAT TabSeparated";
    match query(col_sql).await {
        Ok(body) => {
            let actual: Vec<&str> = body
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect();
            let missing: Vec<&str> = expected_cols
                .iter()
                .filter(|c| !actual.iter().any(|a| a == *c))
                .copied()
                .collect();
            if missing.is_empty() {
                log::info!(
                    "[CH schema] all {}/{} expected columns present",
                    expected_cols.len(),
                    expected_cols.len()
                );
            } else {
                log::warn!(
                    "[CH schema] MISSING columns: {missing:?} — indicators may show no data"
                );
                tg_alert!(
                    crate::telegram::Severity::Warning,
                    "ch-schema",
                    "CH schema missing columns: {missing:?}"
                );
            }
        }
        Err(e) => {
            log::warn!("[CH schema] column check failed: {e}");
        }
    }

    // Parallel check for the forex schema in fxview_cache.forex_bars. Upstream
    // handoff (.planning/HANDOFF-FXVIEW-FOREX-DEFAULTS.md) asked us to verify
    // the contract columns flowsurface actually reads so drift is caught early.
    // Non-fatal: forex is optional; missing the table entirely is fine.
    let forex_expected = [
        "symbol",
        "threshold_decimal_bps",
        "ouroboros_mode",
        "open_time_us",
        "close_time_us",
        "open",
        "high",
        "low",
        "close",
        "quote_count",
        "duration_us",
    ];
    let forex_col_sql = "SELECT name FROM system.columns \
                         WHERE database = 'fxview_cache' AND table = 'forex_bars' \
                         FORMAT TabSeparated";
    if let Ok(body) = query(forex_col_sql).await {
        let actual: Vec<&str> = body
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if actual.is_empty() {
            log::debug!("[CH schema] fxview_cache.forex_bars not found (forex optional)");
        } else {
            let missing: Vec<&str> = forex_expected
                .iter()
                .filter(|c| !actual.iter().any(|a| a == *c))
                .copied()
                .collect();
            if missing.is_empty() {
                log::info!(
                    "[CH schema] forex contract: all {}/{} columns present",
                    forex_expected.len(),
                    forex_expected.len()
                );
            } else {
                log::warn!(
                    "[CH schema] fxview_cache.forex_bars MISSING contract columns: {missing:?}"
                );
                tg_alert!(
                    crate::telegram::Severity::Warning,
                    "ch-schema-forex",
                    "fxview forex_bars missing contract columns: {missing:?}"
                );
            }
        }
    }

    // Query opendeviationbar_version from most recent bar and compare against
    // the compiled crate version to detect sidecar↔app version skew.
    let crate_ver = opendeviationbar_core::Checkpoint::library_version();
    let ver_sql = "SELECT opendeviationbar_version FROM opendeviationbar_cache.open_deviation_bars \
                   ORDER BY close_time_us DESC LIMIT 1 FORMAT TabSeparated";
    match query(ver_sql).await {
        Ok(body) => {
            if let Some(sidecar_ver) = body
                .lines()
                .next()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
            {
                log::info!(
                    "[CH schema] opendeviationbar version: {sidecar_ver} (crate: {crate_ver})"
                );

                // Compare major.minor: strip "+sidecar" suffix and patch version
                let sidecar_major_minor = sidecar_ver
                    .split('+')
                    .next()
                    .and_then(|v| v.rsplit_once('.'))
                    .map(|(mm, _)| mm);
                let crate_major_minor = crate_ver.rsplit_once('.').map(|(mm, _)| mm);

                if sidecar_major_minor != crate_major_minor {
                    log::warn!(
                        "[CH schema] VERSION MISMATCH: sidecar={sidecar_ver} crate={crate_ver} \
                         — rebuild app or update sidecar"
                    );
                    tg_alert!(
                        crate::telegram::Severity::Warning,
                        "odb-version",
                        "ODB version mismatch: sidecar={sidecar_ver} crate={crate_ver} — rebuild app or update sidecar"
                    );
                }
            }
        }
        Err(_) => {
            // opendeviationbar_version column may not exist on older schemas — silently skip
        }
    }
}

/// Returns the ODB symbol allowlist, or None if not yet loaded or empty.
pub fn odb_symbol_filter() -> Option<&'static [String]> {
    ODB_SYMBOLS
        .get()
        .filter(|v| !v.is_empty())
        .map(|v| v.as_slice())
}

fn base_url() -> String {
    APP_CONFIG.base_url()
}

/// Shared HTTP client — reuses connections through the SSH tunnel instead of
/// creating a new TCP handshake per request.
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(2)
        .build()
        .expect("reqwest client build")
});

/// Maximum retries for transient connection/timeout errors.
const CH_MAX_RETRIES: u32 = 3;

pub async fn query(sql: &str) -> Result<String, AdapterError> {
    let url = base_url();
    let sql_preview: String = sql.chars().take(120).collect();
    log::debug!("[CH] POST {url} — {sql_preview}…");

    for attempt in 1..=CH_MAX_RETRIES {
        match HTTP_CLIENT.post(&url).body(sql.to_string()).send().await {
            Ok(resp) => {
                let status = resp.status();
                let content_len = resp.content_length();
                log::debug!(
                    "[CH] response status={status} content_length={content_len:?} sql={sql_preview}…"
                );
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    log::error!("[CH] HTTP {status}: {body} — SQL: {sql_preview}…");
                    let body_preview = &body[..body.len().min(200)];
                    tg_alert!(
                        crate::telegram::Severity::Critical,
                        "clickhouse",
                        "CH HTTP {status}: {body_preview} — SQL: {sql_preview}…"
                    );
                    // HTTP errors (bad SQL, schema mismatch) won't be fixed by retrying
                    return Err(AdapterError::ParseError(format!(
                        "ClickHouse HTTP {}: {}",
                        status, body
                    )));
                }
                return resp.text().await.map_err(|e| {
                    log::error!("[CH] response body read failed: {e}");
                    tg_alert!(
                        crate::telegram::Severity::Warning,
                        "clickhouse",
                        "CH response body read failed"
                    );
                    AdapterError::from(e)
                });
            }
            Err(e) => {
                let retryable = e.is_connect() || e.is_timeout();
                log::warn!(
                    "[CH] attempt {}/{}: {e} (connect={}, timeout={}, retryable={retryable}, url={url})",
                    attempt,
                    CH_MAX_RETRIES,
                    e.is_connect(),
                    e.is_timeout(),
                );
                if !retryable || attempt == CH_MAX_RETRIES {
                    log::error!("[CH] reqwest failed after {attempt} attempt(s): {e} (url={url})");
                    tg_alert!(
                        crate::telegram::Severity::Critical,
                        "clickhouse",
                        "CH request failed after {attempt} attempts: {e} (timeout={}, connect={})",
                        e.is_timeout(),
                        e.is_connect()
                    );
                    return Err(AdapterError::request_failed(
                        &reqwest::Method::POST,
                        &url,
                        e,
                    ));
                }
                // Exponential backoff: 1s, 2s, 4s
                let delay = Duration::from_secs(1 << (attempt - 1));
                log::info!("[CH] retrying in {:?}...", delay);
                tokio::time::sleep(delay).await;
            }
        }
    }
    unreachable!("CH_MAX_RETRIES loop always returns")
}

/// Extract the bare symbol name from a ticker (e.g. "BTCUSDT" from a BinanceLinear ticker).
/// ClickHouse stores symbols without exchange suffixes.
pub fn bare_symbol(ticker_info: &TickerInfo) -> String {
    ticker_info.ticker.to_string()
}

// -- Kline data --

#[derive(Debug, Deserialize, serde::Serialize)]
struct ChKline {
    close_time_us: i64,
    #[serde(default)]
    open_time_us: Option<i64>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    // Forex rows have no trade volume (quote-native schema in fxview_cache); default to 0.
    #[serde(default)]
    buy_volume: f64,
    #[serde(default)]
    sell_volume: f64,
    #[serde(default)]
    individual_trade_count: Option<u32>,
    #[serde(default)]
    ofi: Option<f64>,
    #[serde(default)]
    trade_intensity: Option<f64>,
    #[serde(default)]
    first_agg_trade_id: Option<u64>,
    #[serde(default)]
    last_agg_trade_id: Option<u64>,
    // Gap awareness fields (v13.57+)
    #[serde(default)]
    has_gap: Option<u8>,
    #[serde(default)]
    gap_trade_count: Option<i64>,
    #[serde(default)]
    max_gap_duration_us: Option<i64>,
    #[serde(default)]
    is_exchange_gap: Option<u8>,
    // Tier 5 microstructure (v13.70+)
    #[serde(default)]
    vwap: Option<f64>,
    #[serde(default)]
    duration_us: Option<i64>,
    #[serde(default)]
    is_liquidation_cascade: Option<u8>,
    #[serde(default)]
    vwap_close_deviation: Option<f64>,
    #[serde(default)]
    turnover_imbalance: Option<f64>,
}

pub async fn fetch_klines(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
    range: Option<(u64, u64)>,
) -> Result<Vec<Kline>, AdapterError> {
    let symbol = bare_symbol(&ticker_info);
    let min_tick = ticker_info.min_ticksize;

    let sql = build_odb_sql(&symbol, threshold_dbps, range);

    let body = query(&sql).await?;
    let lines: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let mut klines = Vec::with_capacity(lines.len());

    for line in lines.into_iter().rev() {
        let ck: ChKline = serde_json::from_str(line)
            .map_err(|e| AdapterError::ParseError(format!("ClickHouse kline parse: {e}")))?;

        klines.push(Kline::new(
            (ck.close_time_us / 1000) as u64,
            ck.open as f32,
            ck.high as f32,
            ck.low as f32,
            ck.close as f32,
            Volume::BuySell(
                Qty::from(ck.buy_volume as f32),
                Qty::from(ck.sell_volume as f32),
            ),
            min_tick,
        ));
    }

    Ok(klines)
}

/// Shared SQL builder for ODB queries (includes microstructure columns).
///
/// The initial fetch limit is scaled inversely with threshold so all thresholds
/// show a similar time window. BPR25 (250 dbps) is the reference at 500 bars;
/// BPR50 gets ~250, BPR75 gets ~167. BPR10 gets 13K (floor).
fn build_odb_sql(symbol: &str, threshold_dbps: u32, range: Option<(u64, u64)>) -> String {
    // Forex symbols live in `fxview_cache.forex_bars` (quote-native: bid/ask
    // OHLC, no trade volume / agg ID). Alias quote_count → individual_trade_count
    // so the shared ChKline deserializer works without schema forks.
    // Crypto symbols live in `opendeviationbar_cache.open_deviation_bars`.
    let forex = is_forex_symbol(symbol);
    let (table, cols): (&str, &str) = if forex {
        (
            "fxview_cache.forex_bars",
            "close_time_us, open_time_us, open, high, low, close, \
             quote_count AS individual_trade_count, \
             duration_us",
        )
    } else {
        (
            "opendeviationbar_cache.open_deviation_bars",
            "close_time_us, open_time_us, open, high, low, close, \
             buy_volume, sell_volume, \
             individual_trade_count, ofi, trade_intensity, \
             first_agg_trade_id, last_agg_trade_id, \
             vwap, duration_us, is_liquidation_cascade, \
             vwap_close_deviation, turnover_imbalance",
        )
    };
    // Scale limit inversely with threshold so the visible window stays similar
    // across thresholds. Crypto and forex use different reference points because
    // absolute dbps numbers diverge by 10-50x:
    //   Crypto: 250 dbps (BPR25) is the mid-range reference — 20K bars.
    //   Forex:  25 dbps (BPR2.5) is the mid-range reference — 20K bars.
    // Forex backfill has 2-10M bars per threshold; capping the initial load at
    // ~100K (BPR0.5) keeps payload reasonable while still covering months of
    // data. Floor of 13K fully populates the 7K intensity lookback window.
    let (reference_dbps, reference_limit): (u32, u32) =
        if forex { (25, 20_000) } else { (250, 20_000) };
    let adaptive_limit = ((reference_limit as f64) * (reference_dbps as f64)
        / (threshold_dbps as f64))
        .round()
        .max(13_000.0) as u32;

    // `end == u64::MAX` is the sentinel used by the initial load and sentinel-refetch paths
    // in kline.rs to signal "load the N most recent bars without a time constraint".
    // Scroll-left pagination uses a real `oldest_ts` for `end` and needs the BETWEEN filter.
    let is_full_reload = range.is_none_or(|(_, end)| end == u64::MAX);

    if is_full_reload {
        format!(
            "SELECT {cols} \
             FROM {table} \
             WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold_dbps} \
               AND ouroboros_mode = '{}' \
             ORDER BY close_time_us DESC \
             LIMIT {adaptive_limit} \
             FORMAT JSONEachRow",
            ouroboros_mode_for(symbol)
        )
    } else {
        let (start, end) = range.unwrap();
        // App passes ms timestamps; CH column is in µs — convert at boundary.
        // end_us uses +999 to cover the full millisecond (inclusive upper bound).
        // Without this, bars at close_time_us = end_ms*1000 + 1..999 are missed.
        let start_us = start.saturating_mul(1000);
        let end_us = end.saturating_mul(1000).saturating_add(999);
        format!(
            "SELECT {cols} \
             FROM {table} \
             WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold_dbps} \
               AND ouroboros_mode = '{}' \
               AND close_time_us BETWEEN {start_us} AND {end_us} \
             ORDER BY close_time_us DESC \
             LIMIT 2000 \
             FORMAT JSONEachRow",
            ouroboros_mode_for(symbol)
        )
    }
}

fn parse_microstructure(ck: &ChKline) -> Option<ChMicrostructure> {
    match (ck.individual_trade_count, ck.ofi, ck.trade_intensity) {
        (Some(tc), Some(ofi), Some(ti)) => Some(ChMicrostructure {
            trade_count: tc,
            ofi: ofi as f32,
            trade_intensity: ti as f32,
            has_gap: ck.has_gap.unwrap_or(0) != 0,
            gap_trade_count: ck.gap_trade_count.unwrap_or(0).max(0) as u32,
            vwap: ck.vwap.map(|v| v as f32),
            duration_us: ck.duration_us,
            is_liquidation_cascade: ck.is_liquidation_cascade.unwrap_or(0) != 0,
            vwap_close_deviation: ck.vwap_close_deviation.map(|v| v as f32),
            turnover_imbalance: ck.turnover_imbalance.map(|v| v as f32),
        }),
        _ => None,
    }
}

/// Fetch klines + microstructure sidecar from ClickHouse ODB cache.
pub async fn fetch_klines_with_microstructure(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
    range: Option<(u64, u64)>,
) -> Result<
    (
        Vec<Kline>,
        Vec<Option<ChMicrostructure>>,
        Vec<Option<(u64, u64)>>,
        Vec<Option<u64>>,
    ),
    AdapterError,
> {
    let symbol = bare_symbol(&ticker_info);
    let min_tick = ticker_info.min_ticksize;
    let sql = build_odb_sql(&symbol, threshold_dbps, range);

    let body = query(&sql).await?;
    // Collect non-empty lines first, then iterate in reverse to avoid 4x .reverse() on output Vecs.
    // SQL returns DESC order; reverse-iterating gives ASC (oldest first) directly.
    let lines: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let n = lines.len();
    log::info!(
        "[CH fetch] symbol={symbol} threshold={threshold_dbps} range={range:?} \
         body_len={} lines={n}",
        body.len()
    );
    let mut klines = Vec::with_capacity(n);
    let mut micro = Vec::with_capacity(n);
    let mut agg_id_ranges = Vec::with_capacity(n);
    let mut open_time_ms_list: Vec<Option<u64>> = Vec::with_capacity(n);

    for line in lines.into_iter().rev() {
        let ck: ChKline = serde_json::from_str(line)
            .map_err(|e| AdapterError::ParseError(format!("ClickHouse kline parse: {e}")))?;

        klines.push(Kline::new(
            (ck.close_time_us / 1000) as u64,
            ck.open as f32,
            ck.high as f32,
            ck.low as f32,
            ck.close as f32,
            Volume::BuySell(
                Qty::from(ck.buy_volume as f32),
                Qty::from(ck.sell_volume as f32),
            ),
            min_tick,
        ));
        micro.push(parse_microstructure(&ck));
        agg_id_ranges.push(ck.first_agg_trade_id.zip(ck.last_agg_trade_id));
        open_time_ms_list.push(ck.open_time_us.map(|us| (us / 1000) as u64));
    }

    // Telemetry: log bar statistics for debugging forex rendering
    if !klines.is_empty() {
        let sample = &klines[klines.len().saturating_sub(5)..];
        let body_pcts: Vec<u32> = sample
            .iter()
            .map(|k| {
                let range = k.high.to_f32() - k.low.to_f32();
                let body = (k.close.to_f32() - k.open.to_f32()).abs();
                if range > 0.0 {
                    (body / range * 100.0) as u32
                } else {
                    0
                }
            })
            .collect();
        log::info!(
            "[CH fetch] {} @{} dbps: {} bars, \
             min_tick={:?}, sample body%={:?}, \
             price=[{:.5}..{:.5}]",
            symbol,
            threshold_dbps,
            klines.len(),
            min_tick,
            body_pcts,
            klines.last().map(|k| k.low.to_f32()).unwrap_or(0.0),
            klines.last().map(|k| k.high.to_f32()).unwrap_or(0.0),
        );
    }

    Ok((klines, micro, agg_id_ranges, open_time_ms_list))
}

// -- Backfill request (Issue #97: on-demand trigger for opendeviationbar-py) --

/// Request a backfill by inserting into the backfill_requests table.
/// Returns Ok(true) if the request was inserted, Ok(false) if a recent
/// pending/running request already exists (dedup within 5 minutes).
pub async fn request_backfill(symbol: &str, threshold_dbps: u32) -> Result<bool, AdapterError> {
    // Forex has no backfill path — bars stream live from MT5 via fxview-sidecar;
    // there is no opendeviationbar-py equivalent watching `backfill_requests`.
    if is_forex_symbol(symbol) {
        log::debug!("[CH backfill] skipped for forex symbol {symbol}");
        return Ok(false);
    }
    // Check for recent pending/running request to avoid spam
    let check_sql = format!(
        "SELECT count() as cnt \
         FROM opendeviationbar_cache.backfill_requests FINAL \
         WHERE symbol = '{symbol}' AND status IN ('pending', 'running') \
           AND requested_at > now64(3) - INTERVAL 5 MINUTE \
         FORMAT JSONEachRow"
    );

    let body = query(&check_sql).await?;
    let existing: u64 = body
        .lines()
        .find_map(|line| {
            serde_json::from_str::<serde_json::Value>(line.trim())
                .ok()
                .and_then(|v| v["cnt"].as_u64())
        })
        .unwrap_or(0);

    if existing > 0 {
        log::info!("[CH backfill] request already pending for {symbol}");
        return Ok(false);
    }

    let insert_sql = format!(
        "INSERT INTO opendeviationbar_cache.backfill_requests \
         (symbol, threshold_decimal_bps, source, ouroboros_mode) VALUES \
         ('{symbol}', {threshold_dbps}, 'flowsurface', '{}')",
        ouroboros_mode_for(symbol)
    );

    query(&insert_sql).await?;
    log::info!("[CH backfill] requested backfill for {symbol} @ {threshold_dbps} dbps");
    Ok(true)
}

// -- Streaming (polling) --

pub fn connect_kline_stream(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
) -> impl Stream<Item = Event> {
    // GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/91
    log::info!(
        "[CH poll] connect_kline_stream STARTED: {} @{} dbps",
        ticker_info.ticker,
        threshold_dbps
    );
    connect::channel(16, async move |mut output| {
        let exchange = ticker_info.exchange();
        let _ = output.send(Event::Connected(exchange)).await;

        let stream_kind = StreamKind::OdbKline {
            ticker_info,
            threshold_dbps,
        };

        let symbol = bare_symbol(&ticker_info);

        // Initialize last_ts to the latest bar's timestamp so the first poll
        // doesn't re-fetch bars already loaded by the initial fetch_klines().
        // Retry up to 3 times with 2s backoff — a single transient failure
        // (e.g. SSH tunnel not yet up) would otherwise set last_ts=0, causing
        // the poll loop to crawl from epoch through all historical data.
        // Route watermark query to the correct table (forex → fxview_cache).
        let table = if is_forex_symbol(&symbol) {
            "fxview_cache.forex_bars"
        } else {
            "opendeviationbar_cache.open_deviation_bars"
        };
        let max_ts_sql = format!(
            "SELECT max(close_time_us) AS ts FROM {table} \
             WHERE symbol = '{}' AND threshold_decimal_bps = {} \
               AND ouroboros_mode = '{}' FORMAT JSONEachRow",
            symbol, threshold_dbps, ouroboros_mode_for(&symbol)
        );
        let mut last_ts: u64 = 0;
        for attempt in 1..=3 {
            match query(&max_ts_sql).await {
                Ok(body) => {
                    last_ts = body
                        .lines()
                        .find_map(|line| {
                            serde_json::from_str::<serde_json::Value>(line.trim())
                                .ok()
                                .and_then(|v| v["ts"].as_u64())
                        })
                        .unwrap_or(0);
                    log::info!(
                        "[CH poll] init last_ts={} for {} @{} (attempt {})",
                        last_ts,
                        symbol,
                        threshold_dbps,
                        attempt
                    );
                    break;
                }
                Err(e) => {
                    log::warn!(
                        "[CH poll] init query failed for {} @{} (attempt {}/3): {}",
                        symbol,
                        threshold_dbps,
                        attempt,
                        e
                    );
                    if attempt < 3 {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    } else {
                        tg_alert!(
                            crate::telegram::Severity::Critical,
                            "ch-poll",
                            "CH poll init failed after 3 retries for {symbol}@{threshold_dbps}"
                        );
                    }
                }
            }
        }

        let mut logged_micro_warning = false;

        loop {
            // GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/91
            // 1s polling: evolved from 60s → 5s → 1s as CH load proved trivial
            // and forex (no SSE path) needs the fastest possible push alternative.
            // Each poll is a single-partition SELECT with close_time_us > last_ts
            // — typically <20ms at CH. Further reduction would need real SSE.
            tokio::time::sleep(Duration::from_millis(1000)).await;

            // Forex: quote-native columns aliased to ChKline's trade-centric names.
            // Crypto: full trade-centric column set from opendeviationbar_cache.
            let cols = if is_forex_symbol(&symbol) {
                "close_time_us, open_time_us, open, high, low, close, \
                 quote_count AS individual_trade_count, \
                 duration_us"
            } else {
                "close_time_us, open_time_us, open, high, low, close, \
                 buy_volume, sell_volume, \
                 individual_trade_count, ofi, trade_intensity, \
                 vwap, duration_us, is_liquidation_cascade, \
                 vwap_close_deviation, turnover_imbalance"
            };
            let sql = format!(
                "SELECT {cols} \
                 FROM {table} \
                 WHERE symbol = '{}' AND threshold_decimal_bps = {} \
                   AND ouroboros_mode = '{}' \
                   AND close_time_us > {} \
                 ORDER BY close_time_us ASC \
                 LIMIT 100 \
                 FORMAT JSONEachRow",
                symbol, threshold_dbps, ouroboros_mode_for(&symbol), last_ts
            );

            match query(&sql).await {
                Ok(body) => {
                    let mut count = 0u32;
                    for line in body.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(ck) = serde_json::from_str::<ChKline>(line) {
                            // last_ts stays in µs for SQL WHERE clause comparison
                            let ts_us = ck.close_time_us as u64;
                            if ts_us > last_ts {
                                last_ts = ts_us;
                            }
                            let ts = (ck.close_time_us / 1000) as u64;
                            let raw_f64 = [
                                ck.open,
                                ck.high,
                                ck.low,
                                ck.close,
                                ck.buy_volume,
                                ck.sell_volume,
                            ];
                            let kline = Kline::new(
                                ts,
                                ck.open as f32,
                                ck.high as f32,
                                ck.low as f32,
                                ck.close as f32,
                                Volume::BuySell(
                                    Qty::from(ck.buy_volume as f32),
                                    Qty::from(ck.sell_volume as f32),
                                ),
                                ticker_info.min_ticksize,
                            );
                            let micro = parse_microstructure(&ck);
                            let _ = output
                                .send(Event::KlineReceived(
                                    stream_kind,
                                    kline,
                                    Some(raw_f64),
                                    ck.first_agg_trade_id.zip(ck.last_agg_trade_id),
                                    micro,
                                    ck.open_time_us.map(|us| (us / 1000) as u64),
                                ))
                                .await;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        log::info!(
                            "[CH poll] {} @{}: {} new bars, last_ts={}",
                            symbol,
                            threshold_dbps,
                            count,
                            last_ts
                        );

                        // Defense in depth: if last_ts is >30 days behind now,
                        // the watermark likely started from 0 due to a failed init.
                        // Re-query max(close_time_us) to jump to the present.
                        let now_us = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_micros() as u64)
                            .unwrap_or(0);
                        if last_ts < now_us.saturating_sub(30 * 86_400_000_000) {
                            log::warn!(
                                "[CH poll] {} @{}: last_ts={} is >30 days stale, re-initializing watermark",
                                symbol,
                                threshold_dbps,
                                last_ts
                            );
                            tg_alert!(
                                crate::telegram::Severity::Info,
                                "ch-poll",
                                "CH watermark >30 days stale"
                            );
                            if let Ok(body) = query(&max_ts_sql).await
                                && let Some(ts) = body.lines().find_map(|line| {
                                    serde_json::from_str::<serde_json::Value>(line.trim())
                                        .ok()
                                        .and_then(|v| v["ts"].as_u64())
                                })
                            {
                                last_ts = ts;
                                log::info!(
                                    "[CH poll] {} @{}: watermark reset to {}",
                                    symbol,
                                    threshold_dbps,
                                    last_ts
                                );
                            }
                        }

                        // One-time warning if first polled bar lacks microstructure.
                        // Skip for forex: the fxview schema has no OFI / trade_intensity
                        // (they're trade-centric concepts that don't apply to quote streams).
                        if !logged_micro_warning && !is_forex_symbol(&symbol) {
                            logged_micro_warning = true;
                            if let Some(first_line) = body.lines().find(|l| !l.trim().is_empty())
                                && let Ok(ck) = serde_json::from_str::<ChKline>(first_line.trim())
                                && ck.individual_trade_count.is_none()
                                && ck.ofi.is_none()
                                && ck.trade_intensity.is_none()
                            {
                                log::warn!(
                                    "[CH poll] {} @{}: bars missing microstructure \
                                     — check opendeviationbar-py feature toggles",
                                    symbol,
                                    threshold_dbps
                                );
                                tg_alert!(
                                    crate::telegram::Severity::Info,
                                    "ch-micro",
                                    "CH bars missing microstructure"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    log::warn!(
                        "[CH poll] {} @{}: query error: {}",
                        symbol,
                        threshold_dbps,
                        e
                    );
                    tg_alert!(
                        crate::telegram::Severity::Warning,
                        "ch-poll",
                        "CH poll query error for {symbol}@{threshold_dbps}"
                    );
                }
            }
        }
    })
}

// -- SSE streaming (push-based, replaces polling when enabled) --

static SSE_CONNECTED: AtomicBool = AtomicBool::new(false);

pub fn sse_connected() -> bool {
    SSE_CONNECTED.load(Ordering::Relaxed)
}

pub fn sse_enabled() -> bool {
    APP_CONFIG.sse_enabled
}

fn odb_bar_to_kline_tuple(
    bar: &OdbBar,
    min_tick: MinTicksize,
) -> (Kline, [f64; 6], Option<ChMicrostructure>) {
    let raw_f64 = [
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.buy_volume.unwrap_or(0.0),
        bar.sell_volume.unwrap_or(0.0),
    ];
    let kline = Kline::new(
        (bar.close_time_us / 1000) as u64,
        bar.open as f32,
        bar.high as f32,
        bar.low as f32,
        bar.close as f32,
        Volume::BuySell(
            Qty::from(bar.buy_volume.unwrap_or(0.0) as f32),
            Qty::from(bar.sell_volume.unwrap_or(0.0) as f32),
        ),
        min_tick,
    );
    let micro = match (bar.individual_trade_count, bar.ofi, bar.trade_intensity) {
        (Some(tc), Some(ofi), Some(ti)) => {
            // Gap awareness: OdbBar captures these in `extra` via serde(flatten)
            let has_gap = bar
                .extra
                .get("has_gap")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let gap_trade_count = bar
                .extra
                .get("gap_trade_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            Some(ChMicrostructure {
                trade_count: tc,
                ofi: ofi as f32,
                trade_intensity: ti as f32,
                has_gap,
                gap_trade_count,
                vwap: bar
                    .extra
                    .get("vwap")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                duration_us: bar.extra.get("duration_us").and_then(|v| v.as_i64()),
                is_liquidation_cascade: bar
                    .extra
                    .get("is_liquidation_cascade")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                vwap_close_deviation: bar
                    .extra
                    .get("vwap_close_deviation")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
                turnover_imbalance: bar
                    .extra
                    .get("turnover_imbalance")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32),
            })
        }
        _ => None,
    };
    (kline, raw_f64, micro)
}

pub fn connect_sse_stream(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
) -> impl Stream<Item = Event> {
    log::info!(
        "[SSE] connect_sse_stream STARTED: {} @{} dbps",
        ticker_info.ticker,
        threshold_dbps
    );
    connect::channel(16, async move |mut output| {
        let exchange = ticker_info.exchange();
        let _ = output.send(Event::Connected(exchange)).await;

        let stream_kind = StreamKind::OdbKline {
            ticker_info,
            threshold_dbps,
        };
        let symbol = bare_symbol(&ticker_info);

        let mut attempt: u32 = 0;
        loop {
            attempt += 1;
            log::info!(
                "[SSE] connecting: {} @{} (attempt #{})",
                symbol,
                threshold_dbps,
                attempt
            );

            let client = OdbSseClient::new(OdbSseConfig {
                host: APP_CONFIG.sse_host.clone(),
                port: APP_CONFIG.sse_port,
                symbols: vec![symbol.clone()],
                thresholds: vec![threshold_dbps],
            });

            use futures::StreamExt;
            let mut stream = std::pin::pin!(client.connect());
            while let Some(event) = stream.next().await {
                match event {
                    OdbSseEvent::Connected => {
                        attempt = 0;
                        SSE_CONNECTED.store(true, Ordering::Relaxed);
                        log::info!("[SSE] connected: {} @{}", symbol, threshold_dbps);
                    }
                    OdbSseEvent::Bar(bar) => {
                        if bar.symbol != symbol || bar.threshold != threshold_dbps {
                            continue;
                        }
                        // Skip orphan bars — incomplete bars at UTC midnight boundaries
                        if bar.is_orphan == Some(true) {
                            log::info!("[SSE] skipping orphan bar: ts={}", bar.close_time_us);
                            continue;
                        }
                        let bar_agg_id_range = bar
                            .first_ref_id
                            .filter(|&id| id > 0)
                            .zip(bar.last_ref_id.filter(|&id| id > 0))
                            .map(|(first, last)| (first as u64, last as u64));
                        let (kline, raw_f64, micro) =
                            odb_bar_to_kline_tuple(&bar, ticker_info.min_ticksize);
                        log::info!(
                            "[SSE] {} @{}: bar ts={} agg_id_range={:?}",
                            symbol,
                            threshold_dbps,
                            kline.time,
                            bar_agg_id_range,
                        );
                        let _ = output
                            .send(Event::KlineReceived(
                                stream_kind,
                                kline,
                                Some(raw_f64),
                                bar_agg_id_range,
                                micro,
                                Some((bar.open_time_us / 1000) as u64),
                            ))
                            .await;
                    }
                    OdbSseEvent::Heartbeat => {}
                    OdbSseEvent::DeserializationError { error, raw_data } => {
                        let preview = &raw_data[..raw_data.len().min(120)];
                        log::warn!("[SSE] deser error: {error}, data: {preview}");
                        tg_alert!(
                            crate::telegram::Severity::Warning,
                            "sse",
                            "SSE deser error for {symbol}@{threshold_dbps}: {error} — data: {preview}"
                        );
                    }
                    OdbSseEvent::Disconnected(reason) => {
                        SSE_CONNECTED.store(false, Ordering::Relaxed);
                        log::warn!("[SSE] disconnected: {reason}, reconnecting in 5s");
                        tg_alert!(
                            crate::telegram::Severity::Warning,
                            "sse",
                            "SSE disconnected for {symbol}@{threshold_dbps}: {reason}"
                        );
                        break;
                    }
                    _ => {
                        // FormingBar, Checkpoint, and future variants — ignored for now
                    }
                }
            }
            // Stream ended (with or without Disconnected event)
            SSE_CONNECTED.store(false, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    })
}

// -- Gap-fill: ODB sidecar /catchup endpoint (v12.62.0+) --

/// Binance-compatible gap-fill trade from ODB sidecar.
#[derive(Deserialize)]
struct GapFillTrade {
    #[serde(rename = "a")]
    agg_trade_id: u64,
    #[serde(rename = "T")]
    time: u64,
    #[serde(
        rename = "p",
        deserialize_with = "crate::serde_util::de_string_to_number"
    )]
    price: f32,
    #[serde(
        rename = "q",
        deserialize_with = "crate::serde_util::de_string_to_number"
    )]
    qty: f32,
    #[serde(rename = "m")]
    is_buyer_maker: bool,
}

/// Response from `GET /catchup/{symbol}/{threshold}`.
/// Sidecar handles CH lookup + paginated Parquet+REST internally.
#[derive(Deserialize)]
struct CatchupResponse {
    trades: Vec<GapFillTrade>,
    #[serde(default)]
    through_agg_id: Option<u64>,
    #[serde(default)]
    count: usize,
    #[serde(default)]
    partial: bool,
}

/// Result of the catchup call — trades + fence ID for WS dedup.
pub struct CatchupResult {
    pub trades: Vec<Trade>,
    /// Last agg_trade_id in the catchup range. WS trades <= this are duplicates.
    pub through_agg_id: Option<u64>,
    /// Whether the sidecar returned a partial result (not all trades available).
    pub partial: bool,
    /// Validation warnings collected during post-parse checks.
    pub warnings: Vec<String>,
    /// Client-generated UUID for cross-system log correlation.
    pub request_uuid: uuid::Uuid,
}

/// Post-parse validation of catchup response (Worker-Auditor pattern).
/// ODB sidecar is the worker; flowsurface validates everything before insertion.
fn catchup_response_to_result(catchup: CatchupResponse, request_uuid: uuid::Uuid) -> CatchupResult {
    let mut warnings: Vec<String> = vec![];

    // Check 1: count matches actual trades.len()
    if catchup.count != catchup.trades.len() {
        let msg = format!(
            "trade count mismatch: response claimed {} but contained {}",
            catchup.count,
            catchup.trades.len()
        );
        log::error!("[catchup-validation] uuid={request_uuid}: {msg}");
        tg_alert!(
            crate::telegram::Severity::Critical,
            "catchup-validation",
            "uuid={request_uuid}: {msg}"
        );
        warnings.push(msg);
    }

    // Convert trades
    let trades: Vec<Trade> = catchup
        .trades
        .into_iter()
        .map(|t| Trade {
            time: t.time,
            is_sell: t.is_buyer_maker,
            price: Price::from_f32(t.price),
            qty: Qty::from(t.qty),
            agg_trade_id: Some(t.agg_trade_id),
        })
        .collect();

    // Check 2: through_agg_id matches last trade
    if let (Some(through), Some(last)) = (
        catchup.through_agg_id,
        trades.last().and_then(|t| t.agg_trade_id),
    ) && through != last
    {
        let msg = format!("fence ID mismatch: through_agg_id={through} but last trade={last}");
        log::error!("[catchup-validation] uuid={request_uuid}: {msg}");
        tg_alert!(
            crate::telegram::Severity::Critical,
            "catchup-validation",
            "uuid={request_uuid}: {msg}"
        );
        warnings.push(msg);
    }

    // Check 3: trades strictly ascending by agg_trade_id
    for window in trades.windows(2) {
        if let (Some(a), Some(b)) = (window[0].agg_trade_id, window[1].agg_trade_id)
            && a >= b
        {
            let msg = format!("trades misordered: id {a} >= {b}");
            log::error!("[catchup-validation] uuid={request_uuid}: {msg}");
            tg_alert!(
                crate::telegram::Severity::Critical,
                "catchup-validation",
                "uuid={request_uuid}: {msg}"
            );
            warnings.push(msg);
            break; // only alert once
        }
    }

    // Check 4: internal gaps (warning, not critical — some trades may be non-aggregated)
    let mut internal_gap_count = 0u64;
    for window in trades.windows(2) {
        if let (Some(a), Some(b)) = (window[0].agg_trade_id, window[1].agg_trade_id)
            && b.saturating_sub(a) > 1
        {
            internal_gap_count += b - a - 1;
        }
    }
    if internal_gap_count > 0 {
        let msg = format!("{internal_gap_count} internal gaps in catchup trades");
        log::warn!("[catchup-validation] uuid={request_uuid}: {msg}");
        tg_alert!(
            crate::telegram::Severity::Warning,
            "catchup-validation",
            "uuid={request_uuid}: {msg}"
        );
        warnings.push(msg);
    }

    // Check 5: partial flag
    if catchup.partial {
        let msg = format!("partial response: {} trades, may need retry", trades.len());
        log::warn!("[catchup-validation] uuid={request_uuid}: {msg}");
        tg_alert!(
            crate::telegram::Severity::Warning,
            "catchup-validation",
            "uuid={request_uuid}: {msg}"
        );
        warnings.push(msg);
    }

    CatchupResult {
        trades,
        through_agg_id: catchup.through_agg_id,
        partial: catchup.partial,
        warnings,
        request_uuid,
    }
}

/// Single-call gap-fill from the last committed CH bar to current time.
/// The sidecar (v12.62.0+) handles:
///   1. CH query for last committed bar's last_agg_trade_id
///   2. Paginated Parquet scan (cross-file) + REST bridge
///   3. Rate limiting internally
pub async fn fetch_catchup(
    symbol: &str,
    threshold_dbps: u32,
) -> Result<CatchupResult, AdapterError> {
    let request_uuid = uuid::Uuid::new_v4();
    log::info!("[catchup] {symbol}@{threshold_dbps}: starting, uuid={request_uuid}");

    let url = format!(
        "http://{}:{}/catchup/{symbol}/{threshold_dbps}",
        APP_CONFIG.sse_host, APP_CONFIG.sse_port
    );

    // Retry loop for transient errors from the sidecar:
    // - 429 rate limiting (two panes fire catchup simultaneously)
    // - Transport errors (sidecar drops connection with empty reply under load)
    let mut last_error = None;
    for attempt in 0..3u32 {
        let resp = match HTTP_CLIENT.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                // Transport error (empty reply, connection reset, etc.)
                // Treat as transient and retry, same as 429.
                log::warn!(
                    "[catchup] {symbol}@{threshold_dbps}: transport error, retrying in 5s \
                     (attempt {}/3): {e} (is_timeout={}, is_connect={})",
                    attempt + 1,
                    e.is_timeout(),
                    e.is_connect()
                );
                last_error = Some(format!("catchup transport error: {e}"));
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let body = resp.text().await.unwrap_or_default();
            // Parse "retry after Ns" from response body (e.g. {"error": "rate limited, retry after 5s"})
            let delay_secs = body
                .find("retry after ")
                .and_then(|i| {
                    let rest = &body[i + 12..];
                    rest.trim_end_matches(|c: char| !c.is_ascii_digit())
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse::<u64>()
                        .ok()
                })
                .unwrap_or(5)
                .clamp(1, 30);
            log::info!(
                "[catchup] {symbol}@{threshold_dbps}: 429 rate limited, retrying in {delay_secs}s (attempt {}/3)",
                attempt + 1
            );
            last_error = Some(format!("catchup HTTP 429 Too Many Requests: {body}"));
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            continue;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            log::error!("[catchup] {symbol}@{threshold_dbps}: HTTP {status} — {body}");
            tg_alert!(
                crate::telegram::Severity::Critical,
                "catchup",
                "Catchup HTTP {status}"
            );
            return Err(AdapterError::ParseError(format!(
                "catchup HTTP {status}: {body}"
            )));
        }

        // Success — parse JSON and return
        let catchup: CatchupResponse = resp.json().await?;
        if catchup.partial {
            log::warn!(
                "[catchup] {symbol}@{threshold_dbps}: partial coverage ({} trades)",
                catchup.count
            );
        } else {
            log::info!(
                "[catchup] {symbol}@{threshold_dbps}: {} trades, through_agg_id={:?}",
                catchup.count,
                catchup.through_agg_id
            );
        }

        return Ok(catchup_response_to_result(catchup, request_uuid));
    }

    // All retry attempts exhausted
    Err(AdapterError::ParseError(last_error.unwrap_or_else(|| {
        "catchup: all retry attempts failed".to_string()
    })))
}

// -- Live tick SSE stream (FXView forex) --
//
// Consumer for fxview-sidecar's live tick endpoint.
// Producer contract: `.planning/REPLY-FROM-MQL5-SSE-LIVE-ENDPOINT.md`
// Schema: `GET {FLOWSURFACE_FXVIEW_SSE_URL%/stream}/schema.json`
//
// Invariants enforced here:
// - `time_us` is venue-authoritative; `ring_consumed_at_us` is observability only (never used for bar assembly).
// - `quote_seq` is the idempotency key; survives reconnect via `Last-Event-ID` header.
// - No magic numbers: retry backoff + channel depth are named constants below.
// - Symbol filtering is client-side (sidecar fans out both symbols on one stream).

/// Initial reconnect backoff after first failure.
const FXVIEW_SSE_INITIAL_BACKOFF_S: u64 = 1;
/// Max backoff ceiling (producer's history ring holds ~15 min; 60s keeps us inside).
const FXVIEW_SSE_MAX_BACKOFF_S: u64 = 60;
/// Per-request connect + read timeout. Longer than keepalive so idle keepalive
/// comments don't trip reqwest.
const FXVIEW_SSE_REQUEST_TIMEOUT_S: u64 = 300;
/// mpsc buffer depth for the `connect::channel` subscription — matches
/// `connect_sse_stream` for consistency.
const FXVIEW_SSE_CHANNEL_DEPTH: usize = 64;

#[derive(Debug, Deserialize)]
struct LiveTickEvent {
    symbol: String,
    quote_seq: u64,
    /// Venue-authoritative UTC microseconds. Use this for all bar/chart math.
    time_us: i64,
    /// Sidecar-local receive time (observability only, never use for assembly).
    #[serde(default)]
    #[allow(dead_code)]
    ring_consumed_at_us: i64,
    bid: f64,
    ask: f64,
}

/// Map a live quote event to the shared `Trade` struct.
///
/// Forex quotes are directionless bid/ask updates — we synthesize a Trade from
/// the mid-price so it plugs into the existing last-price-label path. `qty=0`
/// marks it as a zero-volume tick (quote update, not executed trade).
/// `quote_seq` is smuggled through `agg_trade_id` for downstream gap detection.
fn live_tick_to_trade(tick: &LiveTickEvent) -> Trade {
    let mid = ((tick.bid + tick.ask) / 2.0) as f32;
    Trade {
        // time_us → time_ms (venue-authoritative, per handoff invariant)
        time: (tick.time_us / 1000) as u64,
        is_sell: false,
        price: Price::from_f32_lossy(mid),
        qty: Qty::from(0.0_f32),
        agg_trade_id: Some(tick.quote_seq),
    }
}

/// Connect to the fxview-sidecar live tick SSE endpoint and dispatch per-tick
/// `Event::TradesReceived` for each subscribed ticker.
///
/// Handles:
/// - Infinite reconnect with exponential backoff (1s → 60s ceiling).
/// - `Last-Event-ID` replay on reconnect (server buffers ~15 min of history).
/// - Fanout filtering: both symbols arrive on the same stream; we dispatch only
///   for tickers the caller subscribed to.
/// - Bitemporal discipline: only `time_us` (venue-authoritative) feeds the
///   downstream Trade time; `ring_consumed_at_us` is deserialized but unused
///   except for telemetry.
pub fn connect_tick_stream(tickers: Vec<TickerInfo>) -> impl Stream<Item = Event> {
    use futures::StreamExt as _;

    let url = APP_CONFIG.fxview_sse_url.clone();

    connect::channel(FXVIEW_SSE_CHANNEL_DEPTH, async move |mut output| {
        if tickers.is_empty() {
            log::warn!("[fxview-sse] connect_tick_stream called with empty tickers");
            return;
        }

        let exchange = tickers[0].exchange();
        let _ = output.send(Event::Connected(exchange)).await;

        // Symbol → TickerInfo filter for the server's broadcast fanout.
        let symbol_map: rustc_hash::FxHashMap<String, TickerInfo> = tickers
            .iter()
            .map(|ti| (bare_symbol(ti), *ti))
            .collect();
        log::info!(
            "[fxview-sse] subscribing for symbols: {:?}",
            symbol_map.keys().collect::<Vec<_>>()
        );

        let mut last_quote_seq: u64 = 0;
        let mut attempt: u32 = 0;

        // Sample logging: per-symbol tick counter so we can periodically log
        // bid/ask/mid → Trade.price mapping for verification. First tick per
        // symbol is always logged (provability); thereafter every Nth.
        const FXVIEW_SSE_SAMPLE_LOG_EVERY: u64 = 250;
        let mut tick_counts: rustc_hash::FxHashMap<String, u64> =
            rustc_hash::FxHashMap::default();

        loop {
            attempt = attempt.saturating_add(1);
            log::info!(
                "[fxview-sse] connecting (attempt #{attempt}) url={url} last_seq={last_quote_seq}"
            );

            let client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(FXVIEW_SSE_REQUEST_TIMEOUT_S))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    log::error!("[fxview-sse] client build failed: {e}");
                    backoff_sleep(attempt).await;
                    continue;
                }
            };

            let mut req = client
                .get(&url)
                .header(reqwest::header::ACCEPT, "text/event-stream");
            if last_quote_seq > 0 {
                req = req.header("Last-Event-ID", last_quote_seq.to_string());
            }

            let resp = match req.send().await {
                Ok(r) if r.status().is_success() => {
                    log::info!("[fxview-sse] connected, status={}", r.status());
                    attempt = 0;
                    r
                }
                Ok(r) => {
                    log::error!("[fxview-sse] HTTP {}: reconnecting", r.status());
                    tg_alert!(
                        crate::telegram::Severity::Warning,
                        "fxview-sse",
                        "fxview-sse HTTP {status}",
                        status = r.status()
                    );
                    backoff_sleep(attempt).await;
                    continue;
                }
                Err(e) => {
                    log::warn!("[fxview-sse] connect failed: {e}");
                    backoff_sleep(attempt).await;
                    continue;
                }
            };

            // Stream the response body and hand-parse SSE frames.
            // Format: `id: <seq>\n` / `event: tick\n` / `data: <json>\n` / `\n`
            let mut body = resp.bytes_stream();
            let mut buf: Vec<u8> = Vec::with_capacity(4096);
            let mut cur_data: Option<String> = None;

            'read: while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("[fxview-sse] read error: {e}");
                        break 'read;
                    }
                };
                buf.extend_from_slice(&chunk);

                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    // Extract line WITHOUT the trailing \n
                    let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
                    let line = std::str::from_utf8(&line_bytes[..line_bytes.len() - 1])
                        .unwrap_or("")
                        .trim_end_matches('\r');

                    if line.is_empty() {
                        // Frame boundary — dispatch accumulated `data` (if any).
                        if let Some(data_json) = cur_data.take() {
                            match serde_json::from_str::<LiveTickEvent>(&data_json) {
                                Ok(tick) => {
                                    last_quote_seq = tick.quote_seq.max(last_quote_seq);
                                    if let Some(&ticker_info) = symbol_map.get(&tick.symbol) {
                                        let trade = live_tick_to_trade(&tick);
                                        // Provability log: confirm bid/ask/mid → Trade.price
                                        // mapping. First tick per symbol always logged; every
                                        // Nth thereafter to keep telemetry low. Lets the user
                                        // sample-check that the jittering line is the mid.
                                        let n = tick_counts
                                            .entry(tick.symbol.clone())
                                            .and_modify(|c| *c += 1)
                                            .or_insert(1);
                                        if *n == 1 || n.is_multiple_of(FXVIEW_SSE_SAMPLE_LOG_EVERY) {
                                            let mid = (tick.bid + tick.ask) / 2.0;
                                            log::info!(
                                                "[fxview-sse] tick #{n} {symbol} \
                                                 bid={bid} ask={ask} mid={mid:.5} \
                                                 → Trade.price={trade_price:.5} (line uses mid)",
                                                symbol = tick.symbol,
                                                bid = tick.bid,
                                                ask = tick.ask,
                                                trade_price = trade.price.to_f32(),
                                            );
                                        }
                                        let _ = output
                                            .send(Event::TradesReceived(
                                                StreamKind::Trades { ticker_info },
                                                trade.time,
                                                Box::new([trade]),
                                            ))
                                            .await;
                                    }
                                }
                                Err(e) => {
                                    log::warn!(
                                        "[fxview-sse] parse error: {e} data={data_json}"
                                    );
                                }
                            }
                        }
                    } else if let Some(val) = line.strip_prefix("data: ") {
                        cur_data = Some(val.to_string());
                    } else if line.starts_with("id: ")
                        || line.starts_with("event: ")
                        || line.starts_with(':')
                    {
                        // id: and event: fields are handled on frame dispatch;
                        // `:` prefix is an SSE keepalive comment — ignore all.
                    }
                }
            }

            log::warn!("[fxview-sse] stream closed, reconnecting");
            tg_alert!(
                crate::telegram::Severity::Info,
                "fxview-sse",
                "fxview-sse stream closed last_seq={last_quote_seq}"
            );
            backoff_sleep(attempt.saturating_add(1)).await;
        }
    })
}

/// Exponential backoff sleep: 1s, 2s, 4s, …, capped at `FXVIEW_SSE_MAX_BACKOFF_S`.
async fn backoff_sleep(attempt: u32) {
    let shift = attempt.saturating_sub(1).min(6);
    let s = (FXVIEW_SSE_INITIAL_BACKOFF_S << shift).min(FXVIEW_SSE_MAX_BACKOFF_S);
    tokio::time::sleep(Duration::from_secs(s)).await;
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    fn make_gap_fill_trade(id: u64) -> GapFillTrade {
        GapFillTrade {
            agg_trade_id: id,
            time: 1700000000000,
            price: 68500.0,
            qty: 0.001,
            is_buyer_maker: false,
        }
    }

    proptest! {
        #[test]
        fn fence_never_admits_stale_trades(
            fence_id in 1u64..10000,
            trade_ids in prop::collection::vec(1u64..20000, 1..20),
        ) {
            let admitted: Vec<u64> = trade_ids
                .iter()
                .copied()
                .filter(|&id| id > fence_id)
                .collect();
            for id in &admitted {
                prop_assert!(*id > fence_id, "admitted id {} should be > fence {}", id, fence_id);
            }
            // Verify nothing <= fence slipped through
            for id in &trade_ids {
                if *id <= fence_id {
                    prop_assert!(
                        !admitted.contains(id),
                        "stale id {} leaked through fence {}", id, fence_id
                    );
                }
            }
        }

        #[test]
        fn catchup_response_preserves_trade_ids(
            mut ids in prop::collection::vec(1u64..100000, 1..10),
        ) {
            ids.sort();
            ids.dedup();

            let trades: Vec<GapFillTrade> = ids.iter().map(|&id| make_gap_fill_trade(id)).collect();
            let response = CatchupResponse {
                trades,
                through_agg_id: ids.last().copied(),
                count: ids.len(),
                partial: false,
            };

            // tg_alert! inside catchup_response_to_result uses tokio::spawn
            let rt = tokio::runtime::Runtime::new().unwrap();
            let _guard = rt.enter();
            let result = catchup_response_to_result(response, uuid::Uuid::nil());
            let result_ids: Vec<u64> = result
                .trades
                .iter()
                .filter_map(|t| t.agg_trade_id)
                .collect();
            prop_assert_eq!(&result_ids, &ids);
        }

        #[test]
        fn gap_plus_continuity_is_exhaustive(
            prev in 1u64..100000,
            curr in 1u64..100000,
        ) {
            let is_gap = curr.saturating_sub(prev) > 1;
            let is_continuous = curr == prev + 1;
            let is_dup_or_reorder = curr <= prev;

            // Exactly one must be true
            let flags = [is_gap, is_continuous, is_dup_or_reorder];
            let true_count = flags.iter().filter(|&&f| f).count();
            prop_assert_eq!(
                true_count, 1,
                "expected exactly 1 true for prev={}, curr={}: gap={}, cont={}, dup={}",
                prev, curr, is_gap, is_continuous, is_dup_or_reorder
            );
        }
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use insta::assert_json_snapshot;

    fn make_gap_fill_trade(id: u64) -> GapFillTrade {
        GapFillTrade {
            agg_trade_id: id,
            time: 1700000000000,
            price: 68500.0,
            qty: 0.001,
            is_buyer_maker: false,
        }
    }

    #[derive(serde::Serialize)]
    struct CatchupSnapshot {
        trade_count: usize,
        through_agg_id: Option<u64>,
        partial: bool,
        warning_count: usize,
        first_trade_id: Option<u64>,
    }

    impl From<&CatchupResult> for CatchupSnapshot {
        fn from(r: &CatchupResult) -> Self {
            Self {
                trade_count: r.trades.len(),
                through_agg_id: r.through_agg_id,
                partial: r.partial,
                warning_count: r.warnings.len(),
                first_trade_id: r.trades.first().and_then(|t| t.agg_trade_id),
            }
        }
    }

    /// Helper: `catchup_response_to_result` uses `tg_alert!` which calls `tokio::spawn`,
    /// so we need a tokio runtime context even in sync tests.
    fn with_tokio<F: FnOnce() -> R, R>(f: F) -> R {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _guard = rt.enter();
        f()
    }

    #[test]
    fn catchup_response_deserialization() {
        with_tokio(|| {
            let json = r#"{"trades":[{"a":123456,"T":1700000000000,"p":"68500.50","q":"0.001","m":false}],"through_agg_id":123456,"count":1,"partial":false}"#;
            let response: CatchupResponse =
                serde_json::from_str(json).expect("deserialization failed");
            let result = catchup_response_to_result(response, uuid::Uuid::nil());
            assert_json_snapshot!(CatchupSnapshot::from(&result));
        });
    }

    #[test]
    fn catchup_response_partial_flag() {
        with_tokio(|| {
            let response = CatchupResponse {
                trades: vec![],
                through_agg_id: None,
                count: 0,
                partial: true,
            };
            let result = catchup_response_to_result(response, uuid::Uuid::nil());
            assert!(
                result.warnings.iter().any(|w| w.contains("partial")),
                "expected a warning containing 'partial', got: {:?}",
                result.warnings
            );
        });
    }

    #[test]
    fn catchup_response_count_mismatch() {
        with_tokio(|| {
            let response = CatchupResponse {
                trades: vec![make_gap_fill_trade(1)],
                through_agg_id: Some(1),
                count: 5,
                partial: false,
            };
            let result = catchup_response_to_result(response, uuid::Uuid::nil());
            assert!(
                result.warnings.iter().any(|w| w.contains("count mismatch")),
                "expected a warning containing 'count mismatch', got: {:?}",
                result.warnings
            );
        });
    }
}

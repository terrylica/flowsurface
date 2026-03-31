// FILE-SIZE-OK: monolithic adapter — CH HTTP, SSE, catchup, SQL builder are tightly coupled
//! ClickHouse adapter for precomputed open deviation bars from opendeviationbar-py cache.
//!
//! Reads from `opendeviationbar_cache.open_deviation_bars` table via ClickHouse HTTP interface.
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
        agg_trade_id: real_id,
        price: FixedPoint(trade.price.units),
        volume: FixedPoint(trade.qty.units),
        first_trade_id: real_id,
        last_trade_id: real_id,
        timestamp: base_us + sub_ms_offset,
        is_buyer_maker: trade.is_sell,
        is_best_match: None,
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
    }
}

/// ODB symbols fetched from ClickHouse at startup.
/// Populated by `init_odb_symbols()`, accessed synchronously from view code.
static ODB_SYMBOLS: OnceLock<Vec<String>> = OnceLock::new();

/// Fetch available ODB symbols from ClickHouse and cache them.
/// Called once at startup; gracefully returns empty vec on failure.
pub async fn init_odb_symbols() -> Vec<String> {
    let sql = "SELECT DISTINCT symbol FROM opendeviationbar_cache.open_deviation_bars ORDER BY symbol FORMAT TabSeparated";
    match query(sql).await {
        Ok(body) => {
            let symbols: Vec<String> = body
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();
            let count = symbols.len();
            if ODB_SYMBOLS.set(symbols).is_err() {
                log::warn!("ODB symbol cache already initialized");
            } else {
                log::info!("cached {count} ODB symbols from ClickHouse");
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
                if !resp.status().is_success() {
                    let status = resp.status();
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
    buy_volume: f64,
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
    // Both paths use DESC ordering + reverse to get the N most recent bars
    // within the requested window. ASC ordering would return bars from the
    // beginning of time, creating gaps when loading historical data.
    let cols = "close_time_us, open_time_us, open, high, low, close, \
                buy_volume, sell_volume, \
                individual_trade_count, ofi, trade_intensity, \
                first_agg_trade_id, last_agg_trade_id";
    // Filter by ouroboros_mode (default: 'aion'). Aion-mode is the current
    // production mode — continuous bars without UTC-midnight boundaries.
    // Configurable via FLOWSURFACE_OUROBOROS_MODE env var.
    // Scale limit inversely with threshold: BPR25 gets 20,000 bars;
    // all thresholds get a minimum of 13,000 bars to fully populate
    // a 7,000-bar intensity lookback window from the first render.
    let reference_dbps = 250u32;
    let reference_limit = 20_000u32;
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
             FROM opendeviationbar_cache.open_deviation_bars \
             WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold_dbps} \
               AND ouroboros_mode = '{}' \
             ORDER BY close_time_us DESC \
             LIMIT {adaptive_limit} \
             FORMAT JSONEachRow",
            APP_CONFIG.ouroboros_mode
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
             FROM opendeviationbar_cache.open_deviation_bars \
             WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold_dbps} \
               AND ouroboros_mode = '{}' \
               AND close_time_us BETWEEN {start_us} AND {end_us} \
             ORDER BY close_time_us DESC \
             LIMIT 2000 \
             FORMAT JSONEachRow",
            APP_CONFIG.ouroboros_mode
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
            gap_trade_count: ck
                .gap_trade_count
                .unwrap_or(0)
                .max(0) as u32,
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

    Ok((klines, micro, agg_id_ranges, open_time_ms_list))
}

// -- Backfill request (Issue #97: on-demand trigger for opendeviationbar-py) --

/// Request a backfill by inserting into the backfill_requests table.
/// Returns Ok(true) if the request was inserted, Ok(false) if a recent
/// pending/running request already exists (dedup within 5 minutes).
pub async fn request_backfill(symbol: &str, threshold_dbps: u32) -> Result<bool, AdapterError> {
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
        APP_CONFIG.ouroboros_mode
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
        let max_ts_sql = format!(
            "SELECT max(close_time_us) AS ts FROM opendeviationbar_cache.open_deviation_bars \
             WHERE symbol = '{}' AND threshold_decimal_bps = {} \
               AND ouroboros_mode = '{}' FORMAT JSONEachRow",
            symbol, threshold_dbps, APP_CONFIG.ouroboros_mode
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
            // 5s polling for near-real-time ODB bar updates (from 60s)
            tokio::time::sleep(Duration::from_secs(5)).await;

            let sql = format!(
                "SELECT close_time_us, open_time_us, open, high, low, close, \
                        buy_volume, sell_volume, \
                        individual_trade_count, ofi, trade_intensity \
                 FROM opendeviationbar_cache.open_deviation_bars \
                 WHERE symbol = '{}' AND threshold_decimal_bps = {} \
                   AND ouroboros_mode = '{}' \
                   AND close_time_us > {} \
                 ORDER BY close_time_us ASC \
                 LIMIT 100 \
                 FORMAT JSONEachRow",
                symbol, threshold_dbps, APP_CONFIG.ouroboros_mode, last_ts
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

                        // One-time warning if first polled bar lacks microstructure
                        if !logged_micro_warning {
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
                            .first_agg_trade_id
                            .filter(|&id| id > 0)
                            .zip(bar.last_agg_trade_id.filter(|&id| id > 0))
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
    #[serde(rename = "p", deserialize_with = "crate::serde_util::de_string_to_number")]
    price: f32,
    #[serde(rename = "q", deserialize_with = "crate::serde_util::de_string_to_number")]
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

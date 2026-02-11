//! ClickHouse adapter for precomputed range bars from rangebar-py cache.
//!
//! Reads from `rangebar_cache.range_bars` table via ClickHouse HTTP interface.
//! Tickers come from real exchanges (e.g. Binance) — the symbol in ClickHouse
//! is just the base symbol name like "BTCUSDT".
//!
//! Environment variables:
//!   FLOWSURFACE_CH_HOST (default: "bigblack")
//!   FLOWSURFACE_CH_PORT (default: 8123)

use super::{
    super::{Kline, TickerInfo},
    AdapterError, Event, StreamKind,
};

use iced_futures::{
    futures::{SinkExt, Stream},
    stream,
};
use serde::Deserialize;

use std::{sync::LazyLock, time::Duration};

/// Microstructure fields from ClickHouse range bar cache.
/// Kept in exchange crate to avoid circular dependency with data crate.
#[derive(Debug, Clone, Copy)]
pub struct ChMicrostructure {
    pub trade_count: u32,
    pub ofi: f32,
    pub trade_intensity: f32,
}

/// Range bar symbols available in ClickHouse cache.
/// Parsed from FLOWSURFACE_RANGE_BAR_SYMBOLS env var (comma-separated).
/// Empty vec when unset = no filtering (show all tickers).
static RANGE_BAR_SYMBOLS: LazyLock<Vec<String>> = LazyLock::new(|| {
    std::env::var("FLOWSURFACE_RANGE_BAR_SYMBOLS")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').map(|sym| sym.trim().to_string()).collect())
        .unwrap_or_default()
});

/// Returns the range bar symbol allowlist, or None if filtering is disabled.
pub fn range_bar_symbol_filter() -> Option<&'static [String]> {
    let symbols = &*RANGE_BAR_SYMBOLS;
    if symbols.is_empty() {
        None
    } else {
        Some(symbols.as_slice())
    }
}

static CLICKHOUSE_HOST: LazyLock<String> = LazyLock::new(|| {
    std::env::var("FLOWSURFACE_CH_HOST").unwrap_or_else(|_| "bigblack".to_string())
});

static CLICKHOUSE_PORT: LazyLock<u16> = LazyLock::new(|| {
    std::env::var("FLOWSURFACE_CH_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8123)
});

fn base_url() -> String {
    format!("http://{}:{}", *CLICKHOUSE_HOST, *CLICKHOUSE_PORT)
}

async fn query(sql: &str) -> Result<String, AdapterError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(base_url())
        .body(sql.to_string())
        .timeout(Duration::from_secs(30))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AdapterError::ParseError(format!(
            "ClickHouse HTTP {}: {}",
            status, body
        )));
    }

    resp.text().await.map_err(AdapterError::from)
}

/// Extract the bare symbol name from a ticker (e.g. "BTCUSDT" from a BinanceLinear ticker).
/// ClickHouse stores symbols without exchange suffixes.
fn bare_symbol(ticker_info: &TickerInfo) -> String {
    ticker_info.ticker.to_string()
}

// -- Kline data --

#[derive(Debug, Deserialize)]
struct ChKline {
    timestamp_ms: i64,
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
}

pub async fn fetch_klines(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
    range: Option<(u64, u64)>,
) -> Result<Vec<Kline>, AdapterError> {
    let symbol = bare_symbol(&ticker_info);
    let min_tick = ticker_info.min_ticksize;

    let sql = build_range_bar_sql(&symbol, threshold_dbps, range);

    let body = query(&sql).await?;
    let mut klines = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ck: ChKline = serde_json::from_str(line)
            .map_err(|e| AdapterError::ParseError(format!("ClickHouse kline parse: {e}")))?;

        klines.push(Kline::new(
            ck.timestamp_ms as u64,
            ck.open as f32,
            ck.high as f32,
            ck.low as f32,
            ck.close as f32,
            (ck.buy_volume as f32, ck.sell_volume as f32),
            min_tick,
        ));
    }

    // DESC order → reverse to ascending (oldest first)
    klines.reverse();

    Ok(klines)
}

/// Shared SQL builder for range bar queries (includes microstructure columns).
///
/// The initial fetch limit is scaled inversely with threshold so all thresholds
/// show a similar time window. BPR25 (250 dbps) is the reference at 500 bars;
/// BPR50 gets ~250, BPR100 gets ~125.
fn build_range_bar_sql(symbol: &str, threshold_dbps: u32, range: Option<(u64, u64)>) -> String {
    // Both paths use DESC ordering + reverse to get the N most recent bars
    // within the requested window. ASC ordering would return bars from the
    // beginning of time, creating gaps when loading historical data.
    let cols = "timestamp_ms, open, high, low, close, buy_volume, sell_volume, \
                individual_trade_count, ofi, trade_intensity";
    if let Some((start, end)) = range {
        format!(
            "SELECT {cols} \
             FROM rangebar_cache.range_bars \
             WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold_dbps} \
               AND timestamp_ms BETWEEN {start} AND {end} \
             ORDER BY timestamp_ms DESC \
             LIMIT 2000 \
             FORMAT JSONEachRow"
        )
    } else {
        // Scale limit inversely with threshold: 250 dbps → 500, 500 → 250, 1000 → 125
        let reference_dbps = 250u32;
        let reference_limit = 500u32;
        let limit = ((reference_limit as f64) * (reference_dbps as f64) / (threshold_dbps as f64))
            .round()
            .max(100.0) as u32;
        format!(
            "SELECT {cols} \
             FROM rangebar_cache.range_bars \
             WHERE symbol = '{symbol}' AND threshold_decimal_bps = {threshold_dbps} \
             ORDER BY timestamp_ms DESC \
             LIMIT {limit} \
             FORMAT JSONEachRow"
        )
    }
}

fn parse_microstructure(ck: &ChKline) -> Option<ChMicrostructure> {
    match (ck.individual_trade_count, ck.ofi, ck.trade_intensity) {
        (Some(tc), Some(ofi), Some(ti)) => Some(ChMicrostructure {
            trade_count: tc,
            ofi: ofi as f32,
            trade_intensity: ti as f32,
        }),
        _ => None,
    }
}

/// Fetch klines + microstructure sidecar from ClickHouse range bar cache.
pub async fn fetch_klines_with_microstructure(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
    range: Option<(u64, u64)>,
) -> Result<(Vec<Kline>, Vec<Option<ChMicrostructure>>), AdapterError> {
    let symbol = bare_symbol(&ticker_info);
    let min_tick = ticker_info.min_ticksize;
    let sql = build_range_bar_sql(&symbol, threshold_dbps, range);

    let body = query(&sql).await?;
    let mut klines = Vec::new();
    let mut micro = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ck: ChKline = serde_json::from_str(line)
            .map_err(|e| AdapterError::ParseError(format!("ClickHouse kline parse: {e}")))?;

        klines.push(Kline::new(
            ck.timestamp_ms as u64,
            ck.open as f32,
            ck.high as f32,
            ck.low as f32,
            ck.close as f32,
            (ck.buy_volume as f32, ck.sell_volume as f32),
            min_tick,
        ));
        micro.push(parse_microstructure(&ck));
    }

    // DESC order → reverse to ascending (oldest first)
    klines.reverse();
    micro.reverse();

    Ok((klines, micro))
}

// -- Streaming (polling) --

pub fn connect_kline_stream(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
) -> impl Stream<Item = Event> {
    stream::channel(16, async move |mut output| {
        let exchange = ticker_info.exchange();
        let _ = output.send(Event::Connected(exchange)).await;

        let stream_kind = StreamKind::RangeBarKline {
            ticker_info,
            threshold_dbps,
        };

        let symbol = bare_symbol(&ticker_info);

        // Initialize last_ts to the latest bar's timestamp so the first poll
        // doesn't re-fetch bars already loaded by the initial fetch_klines().
        let mut last_ts: u64 = {
            let sql = format!(
                "SELECT max(timestamp_ms) AS ts FROM rangebar_cache.range_bars \
                 WHERE symbol = '{}' AND threshold_decimal_bps = {} FORMAT JSONEachRow",
                symbol, threshold_dbps
            );
            if let Ok(body) = query(&sql).await {
                body.lines()
                    .find_map(|line| {
                        serde_json::from_str::<serde_json::Value>(line.trim())
                            .ok()
                            .and_then(|v| v["ts"].as_u64())
                    })
                    .unwrap_or(0)
            } else {
                0
            }
        };

        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;

            let sql = format!(
                "SELECT timestamp_ms, open, high, low, close, buy_volume, sell_volume, \
                        individual_trade_count, ofi, trade_intensity \
                 FROM rangebar_cache.range_bars \
                 WHERE symbol = '{}' AND threshold_decimal_bps = {} \
                   AND timestamp_ms > {} \
                 ORDER BY timestamp_ms ASC \
                 LIMIT 100 \
                 FORMAT JSONEachRow",
                symbol, threshold_dbps, last_ts
            );

            if let Ok(body) = query(&sql).await {
                for line in body.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if let Ok(ck) = serde_json::from_str::<ChKline>(line) {
                        let ts = ck.timestamp_ms as u64;
                        if ts > last_ts {
                            last_ts = ts;
                        }
                        let kline = Kline::new(
                            ts,
                            ck.open as f32,
                            ck.high as f32,
                            ck.low as f32,
                            ck.close as f32,
                            (ck.buy_volume as f32, ck.sell_volume as f32),
                            ticker_info.min_ticksize,
                        );
                        let _ = output.send(Event::KlineReceived(stream_kind, kline)).await;
                    }
                }
            }
        }
    })
}

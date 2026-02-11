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
        .post(&base_url())
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
}

pub async fn fetch_klines(
    ticker_info: TickerInfo,
    threshold_dbps: u32,
    range: Option<(u64, u64)>,
) -> Result<Vec<Kline>, AdapterError> {
    let symbol = bare_symbol(&ticker_info);
    let min_tick = ticker_info.min_ticksize;

    // Both paths use DESC ordering + reverse to get the N most recent bars
    // within the requested window. ASC ordering would return bars from the
    // beginning of time, creating gaps when loading historical data.
    let sql = if let Some((start, end)) = range {
        format!(
            "SELECT timestamp_ms, open, high, low, close, buy_volume, sell_volume \
             FROM rangebar_cache.range_bars \
             WHERE symbol = '{}' AND threshold_decimal_bps = {} \
               AND timestamp_ms BETWEEN {} AND {} \
             ORDER BY timestamp_ms DESC \
             LIMIT 2000 \
             FORMAT JSONEachRow",
            symbol, threshold_dbps, start, end
        )
    } else {
        format!(
            "SELECT timestamp_ms, open, high, low, close, buy_volume, sell_volume \
             FROM rangebar_cache.range_bars \
             WHERE symbol = '{}' AND threshold_decimal_bps = {} \
             ORDER BY timestamp_ms DESC \
             LIMIT 500 \
             FORMAT JSONEachRow",
            symbol, threshold_dbps
        )
    };

    let body = query(&sql).await?;
    let mut klines = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let ck: ChKline = serde_json::from_str(line).map_err(|e| {
            AdapterError::ParseError(format!("ClickHouse kline parse: {e}"))
        })?;

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
                "SELECT timestamp_ms, open, high, low, close, buy_volume, sell_volume \
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

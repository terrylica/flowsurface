// GitHub Issue: https://github.com/flowsurface-rs/flowsurface/pull/89
use std::time::Duration;

use exchange::unit::{Price, Qty};
use serde::{Deserialize, Serialize};

use crate::util::ok_or_default;

const TRADE_RETENTION_MS: u64 = 120_000;

/// Time display format for trades in the Time & Sales panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimeFormat {
    /// mm:ss.fff (e.g., 45:23.123)
    #[default]
    MinSecMs,
    /// hh:mm:ss (e.g., 14:45:23)
    HourMinSec,
    /// hh:mm:ss.fff (e.g., 14:45:23.123)
    HourMinSecMs,
}

impl TimeFormat {
    pub const ALL: [TimeFormat; 3] = [
        TimeFormat::MinSecMs,
        TimeFormat::HourMinSec,
        TimeFormat::HourMinSecMs,
    ];

    pub fn format_str(&self) -> &'static str {
        match self {
            TimeFormat::MinSecMs => "%M:%S%.3f",
            TimeFormat::HourMinSec => "%H:%M:%S",
            TimeFormat::HourMinSecMs => "%H:%M:%S%.3f",
        }
    }

    /// Formats a timestamp in milliseconds according to the time format and timezone.
    pub fn format_timestamp(&self, ts_ms: u64, timezone: crate::UserTimezone) -> String {
        use chrono::DateTime;

        let Some(datetime) =
            DateTime::from_timestamp(ts_ms as i64 / 1000, (ts_ms % 1000) as u32 * 1_000_000)
        else {
            return String::new();
        };

        let format_str = self.format_str();

        match timezone {
            crate::UserTimezone::Local => datetime
                .with_timezone(&chrono::Local)
                .format(format_str)
                .to_string(),
            crate::UserTimezone::Utc => datetime
                .with_timezone(&chrono::Utc)
                .format(format_str)
                .to_string(),
        }
    }
}

impl std::fmt::Display for TimeFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeFormat::MinSecMs => write!(f, "mm:ss.fff"),
            TimeFormat::HourMinSec => write!(f, "hh:mm:ss"),
            TimeFormat::HourMinSecMs => write!(f, "hh:mm:ss.fff"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub trade_size_filter: f32,
    #[serde(default = "default_buffer_filter")]
    pub trade_retention: Duration,
    #[serde(deserialize_with = "ok_or_default", default)]
    pub stacked_bar: Option<StackedBar>,
    #[serde(deserialize_with = "ok_or_default", default)]
    pub time_format: TimeFormat,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            trade_size_filter: 0.0,
            trade_retention: Duration::from_millis(TRADE_RETENTION_MS),
            stacked_bar: StackedBar::Compact(StackedBarRatio::default()).into(),
            time_format: TimeFormat::default(),
        }
    }
}

fn default_buffer_filter() -> Duration {
    Duration::from_millis(TRADE_RETENTION_MS)
}

#[derive(Debug, Clone)]
pub struct TradeDisplay {
    pub time_str: String,
    pub price: Price,
    pub qty: Qty,
    pub is_sell: bool,
}

#[derive(Debug, Clone)]
pub struct TradeEntry {
    pub ts_ms: u64,
    pub display: TradeDisplay,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Copy)]
pub enum StackedBar {
    Compact(StackedBarRatio),
    Full(StackedBarRatio),
}

impl StackedBar {
    pub fn ratio(self) -> StackedBarRatio {
        match self {
            StackedBar::Compact(r) | StackedBar::Full(r) => r,
        }
    }

    pub fn with_ratio(self, r: StackedBarRatio) -> Self {
        match self {
            StackedBar::Compact(_) => StackedBar::Compact(r),
            StackedBar::Full(_) => StackedBar::Full(r),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Default, Copy)]
pub enum StackedBarRatio {
    Count,
    #[default]
    Volume,
    AverageSize,
}

impl std::fmt::Display for StackedBarRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StackedBarRatio::Count => write!(f, "Count"),
            StackedBarRatio::AverageSize => write!(f, "Average trade size"),
            StackedBarRatio::Volume => write!(f, "Volume"),
        }
    }
}

impl StackedBarRatio {
    pub const ALL: [StackedBarRatio; 3] = [
        StackedBarRatio::Count,
        StackedBarRatio::Volume,
        StackedBarRatio::AverageSize,
    ];
}

#[derive(Debug, Clone, Copy)]
pub enum HistAggValues {
    Count { buy: u64, sell: u64 },
    Qty { buy: Qty, sell: Qty },
}

#[derive(Default)]
pub struct HistAgg {
    buy_count: u64,
    sell_count: u64,
    buy_sum: Qty,
    sell_sum: Qty,
}

impl HistAgg {
    fn average_qty(sum: Qty, count: u64) -> Qty {
        if count == 0 {
            return Qty::ZERO;
        }

        let c = count.min(i64::MAX as u64) as i64;
        let half = c / 2;
        let rounded = if sum.units >= 0 {
            sum.units.saturating_add(half).div_euclid(c)
        } else {
            sum.units.saturating_sub(half).div_euclid(c)
        };

        Qty::from_units(rounded)
    }

    pub fn add(&mut self, trade: &TradeDisplay) {
        let qty = trade.qty;

        if trade.is_sell {
            self.sell_count += 1;
            self.sell_sum += qty;
        } else {
            self.buy_count += 1;
            self.buy_sum += qty;
        }
    }

    pub fn remove(&mut self, trade: &TradeDisplay) {
        let qty = trade.qty;

        if trade.is_sell {
            self.sell_count = self.sell_count.saturating_sub(1);
            self.sell_sum = if self.sell_sum.units >= qty.units {
                self.sell_sum - qty
            } else {
                Qty::ZERO
            };
        } else {
            self.buy_count = self.buy_count.saturating_sub(1);
            self.buy_sum = if self.buy_sum.units >= qty.units {
                self.buy_sum - qty
            } else {
                Qty::ZERO
            };
        }
    }

    pub fn values_for(&self, ratio_kind: StackedBarRatio) -> Option<HistAggValues> {
        match ratio_kind {
            StackedBarRatio::Count => {
                let buy = self.buy_count;
                let sell = self.sell_count;
                let total = buy.saturating_add(sell);

                if total == 0 {
                    return None;
                }

                Some(HistAggValues::Count { buy, sell })
            }
            StackedBarRatio::Volume => {
                let buy = self.buy_sum.to_f32_lossy();
                let sell = self.sell_sum.to_f32_lossy();
                let total = buy + sell;

                if total <= 0.0 {
                    return None;
                }

                Some(HistAggValues::Qty {
                    buy: self.buy_sum,
                    sell: self.sell_sum,
                })
            }
            StackedBarRatio::AverageSize => {
                let buy_avg = Self::average_qty(self.buy_sum, self.buy_count);
                let sell_avg = Self::average_qty(self.sell_sum, self.sell_count);

                let denom = buy_avg.units.saturating_add(sell_avg.units);
                if denom <= 0 {
                    return None;
                }

                Some(HistAggValues::Qty {
                    buy: buy_avg,
                    sell: sell_avg,
                })
            }
        }
    }
}

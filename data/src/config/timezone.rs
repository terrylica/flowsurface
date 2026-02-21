use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::fmt;

// Named format string constants — avoids magic string literals and ensures
// that any future change only needs to happen in one place.
const FMT_TIME_MS: &str = "%M:%S.%3f";
const FMT_DATETIME: &str = "%a %b %-d %H:%M";
const FMT_DATETIME_SEC: &str = "%a %b %-d %H:%M:%S";
const FMT_DATETIME_SEC_MS: &str = "%a %b %-d %H:%M:%S.%3f";

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum UserTimezone {
    #[default]
    Utc,
    Local,
}

impl UserTimezone {
    /// Formats a Unix timestamp (milliseconds) for axis labels based on the selected timezone.
    ///
    /// The input is interpreted as a UTC instant and then converted to either local time
    /// or UTC depending on `self`. The output format is chosen from the `timeframe`:
    ///
    /// - sub-10s intervals: `MM:SS`
    /// - larger intervals at midnight: day of month (`D`)
    /// - otherwise: `HH:MM`
    ///
    /// Returns `Some(formatted_timestamp)` when `timestamp_ms` is valid, otherwise `None`.
    pub fn format_timestamp(
        &self,
        timestamp_ms: i64,
        timeframe: exchange::Timeframe,
    ) -> Option<String> {
        DateTime::from_timestamp_millis(timestamp_ms).map(|datetime| {
            self.with_user_timezone(datetime, |time_with_zone| {
                Self::format_by_timeframe(&time_with_zone, timeframe)
            })
        })
    }

    /// Formats a Unix timestamp (milliseconds) for crosshair tooltips in the selected timezone.
    ///
    /// The input is interpreted as a UTC instant and converted to the user's timezone.
    /// Formatting depends on the provided candle/tick `interval` in milliseconds:
    ///
    /// - sub-10s intervals: `MM:SS.mmm`
    /// - otherwise: `Weekday Mon D HH:MM`
    ///
    /// Returns `Some(formatted_timestamp)` when `timestamp_ms` is valid, otherwise `None`.
    // GitHub Issue: https://github.com/terrylica/flowsurface/issues/1 (upstream-merge: de-duped from String variant)
    pub fn format_crosshair_timestamp(&self, timestamp_ms: i64, interval: u64) -> Option<String> {
        DateTime::from_timestamp_millis(timestamp_ms).map(|datetime| {
            self.with_user_timezone(datetime, |time_with_zone| {
                if interval < 10000 {
                    time_with_zone.format(FMT_TIME_MS).to_string()
                } else {
                    time_with_zone.format(FMT_DATETIME).to_string()
                }
            })
        })
    }

    /// Formats a Unix timestamp (milliseconds) for range bar crosshair tooltips.
    ///
    /// Always includes seconds since range bars can complete within seconds.
    /// Shows milliseconds when `bar_duration_ms` is sub-second (< 1 000 ms).
    pub fn format_range_bar_crosshair(
        &self,
        timestamp_ms: i64,
        bar_duration_ms: u64,
    ) -> Option<String> {
        DateTime::from_timestamp_millis(timestamp_ms).map(|datetime| {
            self.with_user_timezone(datetime, |time_with_zone| {
                if bar_duration_ms < 1_000 {
                    time_with_zone.format(FMT_DATETIME_SEC_MS).to_string()
                } else {
                    time_with_zone.format(FMT_DATETIME_SEC).to_string()
                }
            })
        })
    }

    /// Converts a UTC `DateTime` into the user's configured timezone and normalizes it to
    /// `DateTime<FixedOffset>` so downstream formatting can use one concrete type.
    fn with_user_timezone<T>(
        &self,
        datetime: DateTime<chrono::Utc>,
        formatter: impl FnOnce(DateTime<chrono::FixedOffset>) -> T,
    ) -> T {
        let time_with_zone = match self {
            UserTimezone::Local => datetime.with_timezone(&chrono::Local).fixed_offset(),
            UserTimezone::Utc => datetime.fixed_offset(),
        };

        formatter(time_with_zone)
    }

    /// Formats an already timezone-adjusted timestamp for axis labels.
    ///
    /// `timeframe` controls whether output is second-level (`MM:SS`) or minute-level (`HH:MM`).
    /// At exact midnight for non-sub-10s intervals, this returns the day-of-month (`D`) to
    /// emphasize date boundaries on the chart.
    fn format_by_timeframe(
        datetime: &DateTime<chrono::FixedOffset>,
        timeframe: exchange::Timeframe,
    ) -> String {
        let interval = timeframe.to_milliseconds();

        if interval < 10000 {
            datetime.format("%M:%S").to_string()
        } else if datetime.format("%H:%M").to_string() == "00:00" {
            datetime.format("%-d").to_string()
        } else {
            datetime.format("%H:%M").to_string()
        }
    }

    /// Formats a timestamp for range bar axis labels.
    /// `label_span_ms` is the time span between adjacent labels (not between bars).
    pub fn format_range_bar_label(&self, timestamp: i64, label_span_ms: u64) -> String {
        if let Some(datetime) = DateTime::from_timestamp(timestamp, 0) {
            match self {
                UserTimezone::Local => {
                    let dt = datetime.with_timezone(&chrono::Local);
                    Self::format_range_bar_dt(&dt, label_span_ms)
                }
                UserTimezone::Utc => {
                    let dt = datetime.with_timezone(&chrono::Utc);
                    Self::format_range_bar_dt(&dt, label_span_ms)
                }
            }
        } else {
            String::new()
        }
    }

    // GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97
    fn format_range_bar_dt<Tz: chrono::TimeZone>(
        datetime: &DateTime<Tz>,
        label_span_ms: u64,
    ) -> String
    where
        Tz::Offset: std::fmt::Display,
    {
        if label_span_ms >= 30 * 86_400_000 {
            // Labels > 30 days apart: show month + year
            datetime.format("%b %Y").to_string()
        } else if label_span_ms >= 86_400_000 {
            // Labels > 1 day apart: show month + day + short year
            datetime.format("%b %-d '%y").to_string()
        } else if label_span_ms >= 3_600_000 {
            // Labels > 1 hour apart: show day + short year + time
            datetime.format("%b %-d '%y %H:%M").to_string()
        } else if label_span_ms >= 300_000 {
            // Labels > 5 minutes apart: show HH:MM only
            datetime.format("%H:%M").to_string()
        } else if label_span_ms >= 1_000 {
            // Labels 1s–5min apart: show seconds
            datetime.format("%H:%M:%S").to_string()
        } else {
            // Labels < 1s apart: show milliseconds
            datetime.format("%H:%M:%S.%3f").to_string()
        }
    }

}

impl fmt::Display for UserTimezone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UserTimezone::Utc => write!(f, "UTC"),
            UserTimezone::Local => {
                let local_offset = chrono::Local::now().offset().local_minus_utc();
                let hours = local_offset / 3600;
                let minutes = (local_offset % 3600) / 60;
                write!(f, "Local (UTC {hours:+03}:{minutes:02})")
            }
        }
    }
}

impl<'de> Deserialize<'de> for UserTimezone {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let timezone_str = String::deserialize(deserializer)?;
        match timezone_str.to_lowercase().as_str() {
            "utc" => Ok(UserTimezone::Utc),
            "local" => Ok(UserTimezone::Local),
            _ => Err(serde::de::Error::custom("Invalid UserTimezone")),
        }
    }
}

impl Serialize for UserTimezone {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            UserTimezone::Utc => serializer.serialize_str("UTC"),
            UserTimezone::Local => serializer.serialize_str("Local"),
        }
    }
}

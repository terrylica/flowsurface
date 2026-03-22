//! RSI (Relative Strength Index) indicator powered by kand.
//!
//! Renders as a line subplot with overbought (70) / oversold (30) reference lines.
//! Uses kand's incremental API (`rsi_inc`) for O(1) per-bar updates on live ODB charts.
//! Works on both Time-based and ODB/Tick-based charts.

use crate::chart::{
    Caches, Message, ViewState,
    indicator::{
        indicator_row_slice,
        kline::KlineIndicatorImpl,
        plot::{PlotTooltip, line::LinePlot},
    },
};

use data::chart::{PlotData, kline::KlineDataPoint};
use exchange::{Kline, Trade};

use iced::widget::{center, text};
use kand::ohlcv::rsi::{rsi, rsi_inc};
use std::ops::RangeInclusive;

const DEFAULT_PERIOD: usize = 14;

/// Per-bar RSI data point.
#[derive(Clone, Copy)]
pub struct RsiPoint {
    pub value: f32,
}

/// RSI indicator using kand's batch + incremental computation.
pub struct RsiIndicator {
    cache: Caches,
    /// Forward-indexed storage (0 = oldest bar). Index matches storage_idx.
    data: Vec<RsiPoint>,
    period: usize,
    /// kand RSI state for incremental updates.
    prev_avg_gain: f64,
    prev_avg_loss: f64,
    prev_close: f64,
    /// Number of datapoints processed (for incremental tracking).
    next_idx: usize,
}

impl RsiIndicator {
    pub fn new() -> Self {
        Self {
            cache: Caches::default(),
            data: Vec::new(),
            period: DEFAULT_PERIOD,
            prev_avg_gain: 0.0,
            prev_avg_loss: 0.0,
            prev_close: 0.0,
            next_idx: 0,
        }
    }

    fn reset_state(&mut self) {
        self.data.clear();
        self.prev_avg_gain = 0.0;
        self.prev_avg_loss = 0.0;
        self.prev_close = 0.0;
        self.next_idx = 0;
    }

    /// Batch compute RSI for all bars using kand.
    fn compute_batch(&mut self, closes: &[f64]) {
        self.reset_state();
        let n = closes.len();
        if n < self.period + 1 {
            return;
        }

        let mut rsi_out = vec![0.0_f64; n];
        let mut avg_gain_out = vec![0.0_f64; n];
        let mut avg_loss_out = vec![0.0_f64; n];

        // Use kand batch RSI
        if rsi(closes, self.period, &mut rsi_out, &mut avg_gain_out, &mut avg_loss_out).is_err() {
            return;
        }

        // kand fills output with 0.0 for the first `period` bars (lookback).
        // Store NaN for those to avoid rendering false zero lines.
        let lookback = self.period;
        self.data.reserve(n);
        for (i, &rsi_val) in rsi_out.iter().enumerate() {
            let val = if i < lookback {
                f32::NAN
            } else {
                rsi_val as f32
            };
            self.data.push(RsiPoint { value: val });
        }

        // Save state for incremental updates
        if n > lookback {
            self.prev_avg_gain = avg_gain_out[n - 1];
            self.prev_avg_loss = avg_loss_out[n - 1];
            self.prev_close = closes[n - 1];
        }
        self.next_idx = n;
    }

    /// Incrementally compute RSI for one new bar using kand.
    fn process_one_inc(&mut self, close: f64) {
        if self.prev_close == 0.0 && self.data.is_empty() {
            // Not enough history yet — push NaN
            self.data.push(RsiPoint { value: f32::NAN });
            self.prev_close = close;
            return;
        }

        match rsi_inc(
            close,
            self.prev_close,
            self.prev_avg_gain,
            self.prev_avg_loss,
            self.period,
        ) {
            Ok((rsi_val, new_avg_gain, new_avg_loss)) => {
                self.data.push(RsiPoint {
                    value: rsi_val as f32,
                });
                self.prev_avg_gain = new_avg_gain;
                self.prev_avg_loss = new_avg_loss;
                self.prev_close = close;
            }
            Err(_) => {
                self.data.push(RsiPoint { value: f32::NAN });
                self.prev_close = close;
            }
        }
    }

    fn indicator_elem<'a>(
        &'a self,
        main_chart: &'a ViewState,
        visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        if self.data.is_empty() {
            return center(text("RSI: waiting for data")).into();
        }

        let period = self.period;
        let tooltip = move |p: &RsiPoint, _next: Option<&RsiPoint>| {
            if p.value.is_nan() {
                PlotTooltip::new("RSI: —".to_string())
            } else {
                PlotTooltip::new(format!("RSI({period}): {:.1}", p.value))
            }
        };

        let value_fn = |p: &RsiPoint| p.value;

        let plot = LinePlot::new(value_fn)
            .stroke_width(1.5)
            .padding(0.05)
            .show_points(false)
            .with_tooltip(tooltip);

        indicator_row_slice(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for RsiIndicator {
    fn clear_all_caches(&mut self) {
        self.cache.clear_all();
    }

    fn clear_crosshair_caches(&mut self) {
        self.cache.clear_crosshair();
    }

    fn element<'a>(
        &'a self,
        chart: &'a ViewState,
        visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        self.indicator_elem(chart, visible_range)
    }

    fn rebuild_from_source(&mut self, source: &PlotData<KlineDataPoint>) {
        match source {
            PlotData::TimeBased(timeseries) => {
                let closes: Vec<f64> = timeseries
                    .datapoints
                    .values()
                    .map(|dp| dp.kline.close.to_f32() as f64)
                    .collect();
                self.compute_batch(&closes);
            }
            PlotData::TickBased(tickseries) => {
                let closes: Vec<f64> = tickseries
                    .datapoints
                    .iter()
                    .map(|dp| dp.kline.close.to_f32() as f64)
                    .collect();
                self.compute_batch(&closes);
            }
        }
        self.clear_all_caches();
    }

    fn on_insert_klines(&mut self, _klines: &[Kline]) {}

    fn on_insert_trades(
        &mut self,
        _trades: &[Trade],
        old_dp_len: usize,
        source: &PlotData<KlineDataPoint>,
    ) {
        match source {
            PlotData::TimeBased(_) => return,
            PlotData::TickBased(tickseries) => {
                let new_len = tickseries.datapoints.len();
                if self.next_idx == old_dp_len && old_dp_len > 0 {
                    // Incremental: process only new bars
                    for idx in old_dp_len..new_len {
                        let close = tickseries.datapoints[idx].kline.close.to_f32() as f64;
                        self.process_one_inc(close);
                    }
                    self.next_idx = new_len;
                } else {
                    // State mismatch: full rebuild
                    self.rebuild_from_source(source);
                    return;
                }
            }
        }
        self.clear_all_caches();
    }

    fn on_ticksize_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn on_basis_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

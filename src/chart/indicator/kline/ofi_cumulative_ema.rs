// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97
//! Rolling-window sum of EMA-smoothed OFI. Shows directional accumulation of
//! smoothed order flow over the last N bars (where N = EMA period) as a bar
//! histogram with zero baseline. Shares the OFI EMA period from kline::Config.
//!
//! Unlike a raw cumulative sum (which drifts unboundedly and becomes dominated
//! by ancient history), the rolling window stays responsive to recent order flow
//! direction changes.

use crate::chart::{
    Caches, Message, ViewState,
    indicator::{
        indicator_row,
        kline::KlineIndicatorImpl,
        plot::{
            PlotTooltip,
            bar::{BarClass, BarPlot, Baseline},
        },
    },
};

use data::chart::{PlotData, kline::KlineDataPoint};
use data::conditional_ema::ConditionalEma;
use exchange::{Kline, Trade};

use iced::widget::{center, text};
use std::collections::{BTreeMap, VecDeque};
use std::ops::RangeInclusive;

/// Per-bar data point: rolling sum of EMA(OFI) and candle direction.
#[derive(Clone, Copy)]
struct CumOfiPoint {
    cumsum: f32,
    bullish: bool,
}

/// Cumulative EMA(OFI) indicator using a rolling window sum.
/// Applies EMA to raw OFI, then sums the last `ema_period` EMA values.
pub struct OFICumulativeEmaIndicator {
    cache: Caches,
    data: BTreeMap<u64, CumOfiPoint>,
    ema_period: usize,
}

impl OFICumulativeEmaIndicator {
    pub fn new() -> Self {
        Self::with_ema_period(20)
    }

    pub fn with_ema_period(period: usize) -> Self {
        Self {
            cache: Caches::default(),
            data: BTreeMap::new(),
            ema_period: period,
        }
    }

    /// Compute all data points from scratch using a rolling window.
    fn compute_all(
        datapoints: impl Iterator<Item = (usize, f32, bool)>,
        window_size: usize,
    ) -> BTreeMap<u64, CumOfiPoint> {
        let mut ema_window: VecDeque<f32> = VecDeque::with_capacity(window_size);
        let mut rolling_sum: f32 = 0.0;
        let mut result = BTreeMap::new();

        for (idx, ema_val, bullish) in datapoints {
            // Add new value
            ema_window.push_back(ema_val);
            rolling_sum += ema_val;

            // Evict oldest if window is full
            if ema_window.len() > window_size {
                rolling_sum -= ema_window.pop_front().unwrap_or(0.0);
            }

            result.insert(idx as u64, CumOfiPoint {
                cumsum: rolling_sum,
                bullish,
            });
        }

        result
    }

    fn indicator_elem<'a>(
        &'a self,
        main_chart: &'a ViewState,
        visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        if self.data.is_empty() {
            return center(text("OFI \u{03A3}EMA: no microstructure data")).into();
        }

        let tooltip = |p: &CumOfiPoint, _next: Option<&CumOfiPoint>| {
            let sign = if p.cumsum >= 0.0 { "+" } else { "" };
            PlotTooltip::new(format!("OFI \u{03A3}EMA: {sign}{:.3}", p.cumsum))
        };

        let bar_kind = |p: &CumOfiPoint| BarClass::CandleColored { bullish: p.bullish };
        let value_fn = |p: &CumOfiPoint| p.cumsum;

        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .baseline(Baseline::Zero)
            .with_tooltip(tooltip);

        indicator_row(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for OFICumulativeEmaIndicator {
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
            PlotData::TimeBased(_) => {
                self.data.clear();
            }
            PlotData::TickBased(tickseries) => {
                let mut ema = ConditionalEma::new(self.ema_period);

                let iter = tickseries.datapoints.iter().enumerate().filter_map(|(idx, dp)| {
                    dp.microstructure.map(|m| {
                        let bullish = dp.kline.close >= dp.kline.open;
                        let ema_val = ema.update(m.ofi, true);
                        (idx, ema_val, bullish)
                    })
                });

                self.data = Self::compute_all(iter, self.ema_period);
            }
        }
        self.clear_all_caches();
    }

    fn on_insert_klines(&mut self, _klines: &[Kline]) {}

    fn on_insert_trades(
        &mut self,
        _trades: &[Trade],
        _old_dp_len: usize,
        source: &PlotData<KlineDataPoint>,
    ) {
        // Rolling window requires access to the last N EMA values, so a full
        // rebuild is the simplest correct approach. The computation is O(N)
        // where N = total datapoints, which is fast enough for range bars.
        self.rebuild_from_source(source);
    }

    fn on_ticksize_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn on_basis_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }
}

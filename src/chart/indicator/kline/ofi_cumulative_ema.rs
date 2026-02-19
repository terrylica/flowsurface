// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97
//! Cumulative sum of EMA-smoothed OFI. Shows directional accumulation of
//! smoothed order flow as a bar histogram with zero baseline.
//! Shares the OFI EMA period from kline::Config.

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
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Per-bar data point: cumulative sum of EMA(OFI) and candle direction.
#[derive(Clone, Copy)]
struct CumOfiPoint {
    cumsum: f32,
    bullish: bool,
}

/// Cumulative EMA(OFI) indicator.
/// Applies EMA to raw OFI, then accumulates the running sum.
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
                let mut cumsum: f32 = 0.0;

                self.data = tickseries
                    .datapoints
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, dp)| {
                        dp.microstructure.map(|m| {
                            let bullish = dp.kline.close >= dp.kline.open;
                            let ema_val = ema.update(m.ofi, true);
                            cumsum += ema_val;
                            (idx as u64, CumOfiPoint { cumsum, bullish })
                        })
                    })
                    .collect();
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
        // Full rebuild for cumulative indicator — incremental would need to track
        // the EMA state and cumsum at the last processed index.
        self.rebuild_from_source(source);
    }

    fn on_ticksize_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn on_basis_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }
}

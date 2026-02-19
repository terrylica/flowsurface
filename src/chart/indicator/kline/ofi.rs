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
use exchange::{Kline, Trade};

use iced::widget::{center, text};
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97

/// Order Flow Imbalance indicator: (buy_vol - sell_vol) / total_vol per bar.
/// Range: [-1, 1]. Histogram direction shows +/-, bar color follows candle
/// direction for divergence. Only available for range bars with microstructure.
pub struct OFIIndicator {
    cache: Caches,
    /// (ofi_value, bullish) — bullish = close >= open
    data: BTreeMap<u64, (f32, bool)>,
}

impl OFIIndicator {
    pub fn new() -> Self {
        Self {
            cache: Caches::default(),
            data: BTreeMap::new(),
        }
    }

    fn indicator_elem<'a>(
        &'a self,
        main_chart: &'a ViewState,
        visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        if self.data.is_empty() {
            return center(text("OFI: no microstructure data")).into();
        }

        let tooltip = |value: &(f32, bool), _next: Option<&(f32, bool)>| {
            let sign = if value.0 >= 0.0 { "+" } else { "" };
            PlotTooltip::new(format!("OFI: {sign}{:.3}", value.0))
        };

        let bar_kind = |value: &(f32, bool)| BarClass::CandleColored { bullish: value.1 };

        let value_fn = |v: &(f32, bool)| v.0;

        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .baseline(Baseline::Zero)
            .with_tooltip(tooltip);

        indicator_row(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for OFIIndicator {
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
                self.data = tickseries
                    .datapoints
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, dp)| {
                        dp.microstructure.map(|m| {
                            let bullish = dp.kline.close >= dp.kline.open;
                            (idx as u64, (m.ofi, bullish))
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
        old_dp_len: usize,
        source: &PlotData<KlineDataPoint>,
    ) {
        match source {
            PlotData::TimeBased(_) => return,
            PlotData::TickBased(tickseries) => {
                let start_idx = old_dp_len.saturating_sub(1);
                for (idx, dp) in tickseries.datapoints.iter().enumerate().skip(start_idx) {
                    if let Some(m) = dp.microstructure {
                        let bullish = dp.kline.close >= dp.kline.open;
                        self.data.insert(idx as u64, (m.ofi, bullish));
                    }
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
}

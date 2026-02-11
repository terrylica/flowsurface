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
use data::util::format_with_commas;
use exchange::{Kline, Trade};

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Delta indicator: buy_volume - sell_volume per bar.
/// Positive = buy pressure (green), negative = sell pressure (red).
pub struct DeltaIndicator {
    cache: Caches,
    data: BTreeMap<u64, f32>,
}

impl DeltaIndicator {
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
        let tooltip = |value: &f32, _next: Option<&f32>| {
            let sign = if *value >= 0.0 { "+" } else { "" };
            PlotTooltip::new(format!("Delta: {sign}{}", format_with_commas(*value)))
        };

        let bar_kind = |_value: &f32| BarClass::Signed;

        let value_fn = |v: &f32| *v;

        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .baseline(Baseline::Zero)
            .with_tooltip(tooltip);

        indicator_row(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for DeltaIndicator {
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
                self.data = timeseries
                    .volume_data()
                    .into_iter()
                    .map(|(k, (buy, sell))| (k, buy - sell))
                    .collect();
            }
            PlotData::TickBased(tickseries) => {
                self.data = tickseries.delta_data();
            }
        }
        self.clear_all_caches();
    }

    fn on_insert_klines(&mut self, klines: &[Kline]) {
        for kline in klines {
            self.data
                .insert(kline.time, kline.volume.0 - kline.volume.1);
        }
        self.clear_all_caches();
    }

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
                    self.data
                        .insert(idx as u64, dp.kline.volume.0 - dp.kline.volume.1);
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

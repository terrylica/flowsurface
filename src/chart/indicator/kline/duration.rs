// NOTE(fork): Bar duration indicator — histogram showing bar duration in seconds.
// GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/355

use crate::chart::{
    Caches, Message, ViewState,
    indicator::{
        indicator_row,
        kline::KlineIndicatorImpl,
        plot::{
            PlotTooltip,
            bar::{BarClass, BarPlot},
        },
    },
};

use data::chart::{PlotData, kline::KlineDataPoint};
use exchange::{Kline, Trade};

use iced::widget::{center, text};
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Bar duration indicator: duration_us converted to seconds per bar.
/// Only available for ODB bars with microstructure data from ClickHouse.
pub struct DurationIndicator {
    cache: Caches,
    data: BTreeMap<u64, f32>,
}

impl DurationIndicator {
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
            return center(text("Duration: no microstructure data")).into();
        }

        let tooltip = |value: &f32, _next: Option<&f32>| {
            if *value < 1.0 {
                PlotTooltip::new(format!("Duration: {:.0}ms", value * 1000.0))
            } else if *value < 60.0 {
                PlotTooltip::new(format!("Duration: {:.1}s", value))
            } else {
                PlotTooltip::new(format!(
                    "Duration: {}m {:.0}s",
                    *value as u32 / 60,
                    value % 60.0
                ))
            }
        };

        let bar_kind = |_value: &f32| BarClass::Single;
        let value_fn = |v: &f32| *v;

        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .with_tooltip(tooltip);

        indicator_row(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for DurationIndicator {
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
                self.data = tickseries.duration_data();
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
            PlotData::TimeBased(_) => {}
            PlotData::TickBased(tickseries) => {
                let start_idx = old_dp_len.saturating_sub(1);
                for (idx, dp) in tickseries.datapoints.iter().enumerate().skip(start_idx) {
                    if let Some(m) = dp.microstructure
                        && let Some(dur) = m.duration_us
                    {
                        self.data.insert(idx as u64, dur as f32 / 1_000_000.0);
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

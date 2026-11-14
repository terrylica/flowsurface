// NOTE(fork): Bid-ask spread indicator — per-bar histogram of the quote-native
// spread on forex ODB bars (fxview_cache.forex_bars spread columns).
// GitHub Issue: https://github.com/terrylica/flowsurface/issues/35

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

/// Per-bar spread datapoint: (spread_close, spread_mean, suspect).
///
/// `spread_close` = (ask − bid) at the breach tick; `spread_mean` = mean across
/// all quotes in the bar; `suspect` = producer sanity flag (inverted or
/// implausibly wide spread). Values are in raw price units (e.g. 0.00012 on
/// EURUSD ≈ 1.2 pips).
type SpreadPoint = (f32, f32, bool);

/// Bid-ask spread histogram. Only has data for forex ODB bars — crypto bars
/// are trade-centric and carry no spread columns.
pub struct SpreadIndicator {
    cache: Caches,
    data: BTreeMap<u64, SpreadPoint>,
}

impl SpreadIndicator {
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
            return center(text("Spread: no data (forex ODB bars only)")).into();
        }

        let tooltip = |value: &SpreadPoint, _next: Option<&SpreadPoint>| {
            let (close, mean, suspect) = *value;
            let flag = if suspect { "  ⚠ suspect" } else { "" };
            PlotTooltip::new(format!(
                "Spread close: {close:.5}\nSpread mean: {mean:.5}{flag}"
            ))
        };

        // Suspect bars render sign-colored (danger) by mapping to Signed with
        // a negative value; normal bars are plain histogram bars.
        let bar_kind = |value: &SpreadPoint| {
            if value.2 {
                BarClass::Signed
            } else {
                BarClass::Single
            }
        };
        // Suspect spreads plot negative so the Signed class colors them danger
        // and they visually dip below the axis — an at-a-glance data-quality cue.
        let value_fn = |v: &SpreadPoint| if v.2 { -v.0 } else { v.0 };

        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .with_tooltip(tooltip);

        indicator_row(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for SpreadIndicator {
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
                self.data = tickseries.spread_data();
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
                        && let Some(close) = m.spread_close
                    {
                        self.data.insert(
                            idx as u64,
                            (
                                close,
                                m.spread_mean.unwrap_or(close),
                                m.spread_suspect.unwrap_or(false),
                            ),
                        );
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

// NOTE(fork): VWAP overlay — draws VWAP price line on the main candle chart.
// GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/355

use crate::chart::indicator::kline::KlineIndicatorImpl;
use crate::chart::{Message, ViewState};

use data::chart::PlotData;
use data::chart::kline::KlineDataPoint;
use exchange::unit::Price;
use exchange::{Kline, Trade};

use iced::Color;
use iced::widget::center;
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// VWAP overlay indicator: volume-weighted average price per bar.
/// Draws a continuous line on the main candle chart (no subplot).
/// Only available for ODB bars with microstructure data from ClickHouse.
pub struct VwapOverlayIndicator {
    data: BTreeMap<u64, f32>,
}

impl VwapOverlayIndicator {
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }
}

impl KlineIndicatorImpl for VwapOverlayIndicator {
    fn clear_all_caches(&mut self) {}

    fn clear_crosshair_caches(&mut self) {}

    fn element<'a>(
        &'a self,
        _chart: &'a ViewState,
        _visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        center(iced::widget::text("")).into()
    }

    fn rebuild_from_source(&mut self, source: &PlotData<KlineDataPoint>) {
        match source {
            PlotData::TimeBased(_) => {
                self.data.clear();
            }
            PlotData::TickBased(tickseries) => {
                self.data = tickseries.vwap_data();
            }
        }
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
                        && let Some(vwap) = m.vwap
                    {
                        self.data.insert(idx as u64, vwap);
                    }
                }
            }
        }
    }

    fn on_ticksize_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn on_basis_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn draw_overlay(
        &self,
        frame: &mut iced::widget::canvas::Frame,
        total_len: usize,
        earliest_visual: usize,
        latest_visual: usize,
        price_to_y: &dyn Fn(Price) -> f32,
        interval_to_x: &dyn Fn(u64) -> f32,
        _palette: &iced::theme::palette::Extended,
    ) {
        use iced::widget::canvas::{Stroke, path};

        if self.data.is_empty() {
            return;
        }

        let vwap_color = Color::from_rgb(0.12, 0.56, 1.0); // dodger blue

        let mut builder = path::Builder::new();
        let mut started = false;

        for visual_idx in earliest_visual..=latest_visual {
            let storage_idx = total_len.saturating_sub(1 + visual_idx) as u64;
            if let Some(&vwap) = self.data.get(&storage_idx) {
                let x = interval_to_x(visual_idx as u64);
                let y = price_to_y(Price::from_f32(vwap));
                if started {
                    builder.line_to(iced::Point::new(x, y));
                } else {
                    builder.move_to(iced::Point::new(x, y));
                    started = true;
                }
            }
        }

        if started {
            let path = builder.build();
            frame.stroke(
                &path,
                Stroke::default().with_width(1.5).with_color(vwap_color),
            );
        }
    }
}

// NOTE(fork): Liquidation cascade overlay — marks bars flagged as liquidation cascades.
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

/// Liquidation cascade overlay indicator.
/// Draws diamond markers on bars flagged as liquidation cascades.
/// Only available for ODB bars with microstructure data from ClickHouse.
pub struct LiquidationCascadeIndicator {
    data: BTreeMap<u64, bool>,
}

impl LiquidationCascadeIndicator {
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }
}

impl KlineIndicatorImpl for LiquidationCascadeIndicator {
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
                self.data = tickseries.liquidation_cascade_data();
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
                    if let Some(m) = dp.microstructure {
                        self.data.insert(idx as u64, m.is_liquidation_cascade);
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
        use iced::widget::canvas::path;

        if self.data.is_empty() {
            return;
        }

        // Orange-red for liquidation cascade markers
        let marker_color = Color::from_rgb(1.0, 0.35, 0.1);
        let half_size = 4.0_f32;

        for visual_idx in earliest_visual..=latest_visual {
            let storage_idx = total_len.saturating_sub(1 + visual_idx) as u64;
            if self.data.get(&storage_idx) != Some(&true) {
                continue;
            }
            let x = interval_to_x(visual_idx as u64);
            // Place marker above the bar area (use a high price offset)
            let y = price_to_y(Price::from_units(0)) - 8.0;
            // Fallback: draw at top of visible frame area
            let y = if y.is_finite() { y } else { 10.0 };

            // Diamond shape: 4 points
            let mut builder = path::Builder::new();
            builder.move_to(iced::Point::new(x, y - half_size));
            builder.line_to(iced::Point::new(x + half_size, y));
            builder.line_to(iced::Point::new(x, y + half_size));
            builder.line_to(iced::Point::new(x - half_size, y));
            builder.close();

            let diamond = builder.build();
            frame.fill(&diamond, marker_color);
        }
    }
}

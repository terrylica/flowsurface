//! Shared chart overlay drawing utilities.
//!
//! This module groups screen-space drawing functions used across chart types:
//! - **Watermark**: version + build info (bottom-left corner)
//! - **Volume bar**: buy/sell proportion bar (footprint cells, volume indicator)
//!
//! Chart-type-specific legends remain in their respective modules:
//! - Crosshair tooltip → `kline/crosshair.rs`
//! - Bar selection stats → `kline/bar_selection.rs`
//! - Thermal color scale → `indicator/kline/trade_intensity_heatmap.rs`
//! - Depth heatmap tooltip → `heatmap.rs`

use crate::style;

use iced::theme::palette::Extended;
use iced::widget::canvas::{self, Cache, Geometry};
use iced::{Alignment, Point, Renderer, Size};

/// Draw the version watermark at the bottom-left corner of the chart.
pub fn draw_watermark(
    cache: &Cache,
    renderer: &Renderer,
    bounds_size: Size,
    palette: &Extended,
) -> Geometry {
    cache.draw(renderer, bounds_size, |frame| {
        let content = format!(
            "v{} \u{00B7} ODB {} \u{00B7} {}",
            env!("CARGO_PKG_VERSION"),
            env!("FLOWSURFACE_ODB_VERSION"),
            env!("FLOWSURFACE_BUILD_TIME"),
        );
        frame.fill_text(canvas::Text {
            content,
            position: Point::new(8.0, bounds_size.height - 8.0),
            size: iced::Pixels(13.0),
            color: palette.background.base.text.scale_alpha(0.3),
            font: style::AZERET_MONO,
            align_x: Alignment::Start.into(),
            align_y: Alignment::End.into(),
            ..canvas::Text::default()
        });
    })
}

/// Draw a proportional buy/sell volume bar.
///
/// Used by footprint rendering and volume indicators. Renders a stacked bar
/// with sell portion first (closer to axis), buy portion second.
pub(crate) fn draw_volume_bar(
    frame: &mut canvas::Frame,
    start_x: f32,
    start_y: f32,
    buy_qty: f32,
    sell_qty: f32,
    max_qty: f32,
    bar_length: f32,
    thickness: f32,
    buy_color: iced::Color,
    sell_color: iced::Color,
    bar_color_alpha: f32,
    horizontal: bool,
) {
    let total_qty = buy_qty + sell_qty;
    if total_qty <= 0.0 || max_qty <= 0.0 {
        return;
    }

    let total_bar_length = (total_qty / max_qty) * bar_length;

    let buy_proportion = buy_qty / total_qty;
    let sell_proportion = sell_qty / total_qty;

    let buy_bar_length = buy_proportion * total_bar_length;
    let sell_bar_length = sell_proportion * total_bar_length;

    if horizontal {
        let start_y = start_y - (thickness / 2.0);

        if sell_qty > 0.0 {
            frame.fill_rectangle(
                Point::new(start_x, start_y),
                Size::new(sell_bar_length, thickness),
                sell_color.scale_alpha(bar_color_alpha),
            );
        }

        if buy_qty > 0.0 {
            frame.fill_rectangle(
                Point::new(start_x + sell_bar_length, start_y),
                Size::new(buy_bar_length, thickness),
                buy_color.scale_alpha(bar_color_alpha),
            );
        }
    } else {
        let start_x = start_x - (thickness / 2.0);

        if sell_qty > 0.0 {
            frame.fill_rectangle(
                Point::new(start_x, start_y + (bar_length - sell_bar_length)),
                Size::new(thickness, sell_bar_length),
                sell_color.scale_alpha(bar_color_alpha),
            );
        }

        if buy_qty > 0.0 {
            frame.fill_rectangle(
                Point::new(
                    start_x,
                    start_y + (bar_length - sell_bar_length - buy_bar_length),
                ),
                Size::new(thickness, buy_bar_length),
                buy_color.scale_alpha(bar_color_alpha),
            );
        }
    }
}

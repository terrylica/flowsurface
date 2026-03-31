use crate::chart::interaction::{Interaction, TEXT_SIZE};
use crate::style;
use data::chart::{Basis, ViewConfig};
use exchange::TickerInfo;
use exchange::unit::{Price, PriceStep};

use iced::theme::palette::Extended;
use iced::widget::canvas::{self, Cache, Frame, LineDash, Path, Stroke};
use iced::{Alignment, Length, Point, Rectangle, Size};

use super::scale::linear::PriceInfoLabel;

pub trait PlotConstants {
    fn min_scaling(&self) -> f32;
    fn max_scaling(&self) -> f32;
    fn max_cell_width(&self) -> f32;
    fn min_cell_width(&self) -> f32;
    fn max_cell_height(&self) -> f32;
    fn min_cell_height(&self) -> f32;
    fn default_cell_width(&self) -> f32;
}

#[derive(Default)]
pub struct Caches {
    pub(crate) main: Cache,
    pub(crate) x_labels: Cache,
    pub(crate) y_labels: Cache,
    pub(crate) crosshair: Cache,
    pub(crate) watermark: Cache,
    /// Screen-space indicator legend overlay (e.g. thermal colour scale).
    /// Cleared by `clear_all` only — not invalidated on crosshair moves.
    pub legend: Cache,
}

impl Caches {
    pub(crate) fn clear_all(&self) {
        self.main.clear();
        self.x_labels.clear();
        self.y_labels.clear();
        self.crosshair.clear();
        self.watermark.clear();
        self.legend.clear();
    }

    pub(crate) fn clear_crosshair(&self) {
        self.crosshair.clear();
        self.y_labels.clear();
        self.x_labels.clear();
    }

    /// Returns the watermark cache for reuse as an indicator subplot legend layer.
    /// Unused in the indicator subplot pipeline (watermark is only drawn on the main chart).
    pub fn legend_cache(&self) -> &Cache {
        &self.watermark
    }
}

pub struct ViewState {
    pub(crate) cache: Caches,
    pub(crate) bounds: Rectangle,
    pub(crate) translation: iced::Vector,
    pub(crate) scaling: f32,
    pub(crate) cell_width: f32,
    pub(crate) cell_height: f32,
    pub(crate) basis: Basis,
    pub(crate) last_price: Option<PriceInfoLabel>,
    pub(crate) last_trade_time: Option<u64>,
    pub(crate) base_price_y: Price,
    pub(crate) latest_x: u64,
    pub(crate) tick_size: PriceStep,
    pub(crate) decimals: usize,
    pub(crate) ticker_info: TickerInfo,
    pub(crate) layout: ViewConfig,
    // GitHub Issue: https://github.com/terrylica/flowsurface/issues/2
    // Cell allows the view() fn to write timezone through &self so canvas::Program::draw
    // can read it without needing &mut access or an extra parameter.
    pub(crate) timezone: std::cell::Cell<data::UserTimezone>,
    /// Timestamp of last zoom-triggered invalidation.
    pub(crate) last_zoom_invalidation: std::time::Instant,
    /// True if a zoom state change was applied but invalidation was throttled.
    pub(crate) zoom_pending_redraw: bool,
    /// EMA-smoothed render duration from the last draw() cycle (microseconds).
    /// Cell for interior mutability — updated from draw(&self).
    /// Used as the adaptive throttle interval — no hardcoded timing constants.
    pub(crate) frame_budget_us: std::cell::Cell<f32>,
}

impl ViewState {
    pub fn new(
        basis: Basis,
        tick_size: PriceStep,
        decimals: usize,
        ticker_info: TickerInfo,
        layout: ViewConfig,
        cell_width: f32,
        cell_height: f32,
    ) -> Self {
        ViewState {
            cache: Caches::default(),
            bounds: Rectangle::default(),
            translation: iced::Vector::default(),
            scaling: 1.0,
            cell_width,
            cell_height,
            basis,
            last_price: None,
            last_trade_time: None,
            base_price_y: Price::from_f32_lossy(0.0),
            latest_x: 0,
            tick_size,
            decimals,
            ticker_info,
            layout,
            timezone: std::cell::Cell::new(data::UserTimezone::Utc),
            last_zoom_invalidation: std::time::Instant::now(),
            zoom_pending_redraw: false,
            frame_budget_us: std::cell::Cell::new(8000.0), // 8ms — converges after first draw
        }
    }

    #[inline]
    pub(crate) fn price_unit() -> i64 {
        // Price atomic scale is 10^-8 (8 decimal places)
        10i64.pow(8)
    }

    pub(crate) fn visible_region(&self, size: Size) -> Rectangle {
        let width = size.width / self.scaling;
        let height = size.height / self.scaling;

        Rectangle {
            x: -self.translation.x - width / 2.0,
            y: -self.translation.y - height / 2.0,
            width,
            height,
        }
    }

    pub(crate) fn is_interval_x_visible(&self, interval_x: f32) -> bool {
        let region = self.visible_region(self.bounds.size());

        interval_x >= region.x && interval_x <= region.x + region.width
    }

    pub(crate) fn interval_range(&self, region: &Rectangle) -> (u64, u64) {
        match self.basis {
            Basis::Tick(_) | Basis::Odb(_) => (
                self.x_to_interval(region.x + region.width),
                self.x_to_interval(region.x),
            ),
            Basis::Time(timeframe) => {
                let interval = timeframe.to_milliseconds();
                (
                    self.x_to_interval(region.x).saturating_sub(interval / 2),
                    self.x_to_interval(region.x + region.width)
                        .saturating_add(interval / 2),
                )
            }
        }
    }

    pub(crate) fn price_range(&self, region: &Rectangle) -> (Price, Price) {
        let highest = self.y_to_price(region.y);
        let lowest = self.y_to_price(region.y + region.height);

        (highest, lowest)
    }

    pub(crate) fn interval_to_x(&self, value: u64) -> f32 {
        match self.basis {
            Basis::Time(timeframe) => {
                let interval = timeframe.to_milliseconds() as f64;
                let cell_width = f64::from(self.cell_width);

                let diff = value as f64 - self.latest_x as f64;
                (diff / interval * cell_width) as f32
            }
            Basis::Tick(_) | Basis::Odb(_) => -((value as f32) * self.cell_width),
        }
    }

    pub(crate) fn x_to_interval(&self, x: f32) -> u64 {
        match self.basis {
            Basis::Time(timeframe) => {
                let interval = timeframe.to_milliseconds();

                if x <= 0.0 {
                    let diff = (-x / self.cell_width * interval as f32) as u64;
                    self.latest_x.saturating_sub(diff)
                } else {
                    let diff = (x / self.cell_width * interval as f32) as u64;
                    self.latest_x.saturating_add(diff)
                }
            }
            Basis::Tick(_) | Basis::Odb(_) => {
                let tick = -(x / self.cell_width);
                tick.round() as u64
            }
        }
    }

    pub(crate) fn price_to_y(&self, price: Price) -> f32 {
        let result = if self.tick_size.units == 0 {
            let one = Self::price_unit() as f32;
            let delta_units = (self.base_price_y.units - price.units) as f32;
            (delta_units / one) * self.cell_height
        } else {
            let delta_units = self.base_price_y.units - price.units;
            let ticks = (delta_units as f32) / (self.tick_size.units as f32);
            ticks * self.cell_height
        };

        if !result.is_finite() {
            return 0.0;
        }
        result
    }

    pub(crate) fn y_to_price(&self, y: f32) -> Price {
        if self.tick_size.units == 0 {
            let one = Self::price_unit() as f32;
            let delta_units = ((y / self.cell_height) * one).round() as i64;
            return Price::from_units(self.base_price_y.units - delta_units);
        }

        let ticks: f32 = y / self.cell_height;
        let delta_units = (ticks * self.tick_size.units as f32).round() as i64;
        Price::from_units(self.base_price_y.units - delta_units)
    }

    pub(crate) fn draw_crosshair(
        &self,
        frame: &mut Frame,
        theme: &iced::Theme,
        bounds: Size,
        cursor_position: Point,
        interaction: &Interaction,
    ) -> (f32, u64) {
        let region = self.visible_region(bounds);
        let dashed_line = style::dashed_line(theme);

        let highest_p: Price = self.y_to_price(region.y);
        let lowest_p: Price = self.y_to_price(region.y + region.height);
        let highest: f32 = highest_p.to_f32_lossy();
        let lowest: f32 = lowest_p.to_f32_lossy();

        let tick_size = self.tick_size.to_f32_lossy();

        if let Interaction::Ruler { start: Some(start) } = interaction {
            let p1 = *start;
            let p2 = cursor_position;

            let snap_y = |y: f32| {
                let ratio = y / bounds.height;
                let price = highest + ratio * (lowest - highest);
                let price_range = lowest - highest;

                let rounded_price_p = if self.tick_size.units == 0 {
                    if tick_size > 0.0 {
                        Price::from_f32_lossy((price / tick_size).round() * tick_size)
                    } else {
                        Price::from_f32_lossy(price)
                    }
                } else {
                    let p = Price::from_f32_lossy(price);
                    let tick_units = self.tick_size.units;
                    let tick_index = p.units.div_euclid(tick_units);
                    Price::from_units(tick_index * tick_units)
                };
                let rounded_price = rounded_price_p.to_f32_lossy();
                let snap_ratio = if price_range.abs() > f32::EPSILON {
                    (rounded_price - highest) / price_range
                } else {
                    0.5
                };
                snap_ratio * bounds.height
            };

            let snap_x = |x: f32| {
                let (_, snap_ratio) = self.snap_x_to_index(x, bounds, region);
                snap_ratio * bounds.width
            };

            let snapped_p1_x = snap_x(p1.x);
            let snapped_p1_y = snap_y(p1.y);
            let snapped_p2_x = snap_x(p2.x);
            let snapped_p2_y = snap_y(p2.y);

            let price1 = self.y_to_price(snapped_p1_y);
            let price2 = self.y_to_price(snapped_p2_y);

            let pct = if price1.to_f32_lossy() == 0.0 {
                0.0
            } else {
                ((price2.to_f32_lossy() - price1.to_f32_lossy()) / price1.to_f32_lossy()) * 100.0
            };
            let pct_text = format!("{:.2}%", pct);

            let interval_diff: String = match self.basis {
                Basis::Time(_) => {
                    let (timestamp1, _) = self.snap_x_to_index(p1.x, bounds, region);
                    let (timestamp2, _) = self.snap_x_to_index(p2.x, bounds, region);

                    let diff_ms: u64 = timestamp1.abs_diff(timestamp2);
                    data::util::format_duration_ms(diff_ms)
                }
                Basis::Tick(_) => {
                    let (tick1, _) = self.snap_x_to_index(p1.x, bounds, region);
                    let (tick2, _) = self.snap_x_to_index(p2.x, bounds, region);

                    let tick_diff = tick1.abs_diff(tick2);
                    format!("{} ticks", tick_diff)
                }
                Basis::Odb(_) => {
                    let (idx1, _) = self.snap_x_to_index(p1.x, bounds, region);
                    let (idx2, _) = self.snap_x_to_index(p2.x, bounds, region);

                    let bar_diff = idx1.abs_diff(idx2);
                    format!("{} bars", bar_diff)
                }
            };

            let rect_x = snapped_p1_x.min(snapped_p2_x);
            let rect_y = snapped_p1_y.min(snapped_p2_y);
            let rect_w = (snapped_p1_x - snapped_p2_x).abs();
            let rect_h = (snapped_p1_y - snapped_p2_y).abs();

            let palette = theme.extended_palette();

            frame.fill_rectangle(
                Point::new(rect_x, rect_y),
                Size::new(rect_w, rect_h),
                palette.primary.base.color.scale_alpha(0.08),
            );
            let corners = [
                Point::new(rect_x, rect_y),
                Point::new(rect_x + rect_w, rect_y),
                Point::new(rect_x, rect_y + rect_h),
                Point::new(rect_x + rect_w, rect_y + rect_h),
            ];

            let (text_corner, idx) = corners
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.x - p2.x).hypot(a.y - p2.y);
                    let db = (b.x - p2.x).hypot(b.y - p2.y);
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(i, &c)| (c, i))
                .unwrap();

            let text_padding = 8.0;
            let text_pos = match idx {
                0 => Point::new(text_corner.x + text_padding, text_corner.y + text_padding),
                1 => Point::new(text_corner.x - text_padding, text_corner.y + text_padding),
                2 => Point::new(text_corner.x + text_padding, text_corner.y - text_padding),
                3 => Point::new(text_corner.x - text_padding, text_corner.y - text_padding),
                _ => text_corner,
            };

            let datapoints_text = match self.basis {
                Basis::Time(timeframe) => {
                    let interval_ms = timeframe.to_milliseconds();
                    let (timestamp1, _) = self.snap_x_to_index(p1.x, bounds, region);
                    let (timestamp2, _) = self.snap_x_to_index(p2.x, bounds, region);

                    let diff_ms = timestamp1.abs_diff(timestamp2);
                    let datapoints = (diff_ms / interval_ms).max(1);
                    format!("{} bars", datapoints)
                }
                Basis::Tick(aggregation) => {
                    let (tick1, _) = self.snap_x_to_index(p1.x, bounds, region);
                    let (tick2, _) = self.snap_x_to_index(p2.x, bounds, region);

                    let tick_diff = tick1.abs_diff(tick2);
                    let datapoints = (tick_diff / u64::from(aggregation.0)).max(1);
                    format!("{} bars", datapoints)
                }
                Basis::Odb(_) => {
                    let (idx1, _) = self.snap_x_to_index(p1.x, bounds, region);
                    let (idx2, _) = self.snap_x_to_index(p2.x, bounds, region);

                    let bar_diff = idx1.abs_diff(idx2).max(1);
                    format!("{} bars", bar_diff)
                }
            };

            let label_text = format!("{}, {} | {}", datapoints_text, interval_diff, pct_text);

            let text_width = (label_text.len() as f32) * TEXT_SIZE * 0.6;
            let text_height = TEXT_SIZE * 1.2;
            let rect_padding = 4.0;

            let (bg_x, bg_y) = match idx {
                0 => (text_pos.x - rect_padding, text_pos.y - rect_padding),
                1 => (
                    text_pos.x - text_width - rect_padding,
                    text_pos.y - rect_padding,
                ),
                2 => (
                    text_pos.x - rect_padding,
                    text_pos.y - text_height - rect_padding,
                ),
                3 => (
                    text_pos.x - text_width - rect_padding,
                    text_pos.y - text_height - rect_padding,
                ),
                _ => (
                    text_pos.x - text_width / 2.0 - rect_padding,
                    text_pos.y - text_height / 2.0 - rect_padding,
                ),
            };

            frame.fill_rectangle(
                Point::new(bg_x, bg_y),
                Size::new(
                    text_width + rect_padding * 2.0,
                    text_height + rect_padding * 2.0,
                ),
                palette.background.weakest.color.scale_alpha(0.9),
            );

            frame.fill_text(iced::widget::canvas::Text {
                content: label_text,
                position: text_pos,
                color: palette.background.base.text,
                size: iced::Pixels(11.0),
                align_x: match idx {
                    0 | 2 => Alignment::Start.into(),
                    1 | 3 => Alignment::End.into(),
                    _ => Alignment::Center.into(),
                },
                align_y: match idx {
                    0 | 1 => Alignment::Start.into(),
                    2 | 3 => Alignment::End.into(),
                    _ => Alignment::Center.into(),
                },
                font: style::AZERET_MONO,
                ..Default::default()
            });
        }

        // Horizontal price line
        let crosshair_ratio = cursor_position.y / bounds.height;
        let crosshair_price = highest + crosshair_ratio * (lowest - highest);

        let price_range = lowest - highest;
        let rounded_price = if tick_size > 0.0 {
            (crosshair_price / tick_size).round() * tick_size
        } else {
            crosshair_price
        };
        let snap_ratio = if price_range.abs() > f32::EPSILON {
            (rounded_price - highest) / price_range
        } else {
            0.5
        };

        if snap_ratio.is_finite() {
            frame.stroke(
                &Path::line(
                    Point::new(0.0, snap_ratio * bounds.height),
                    Point::new(bounds.width, snap_ratio * bounds.height),
                ),
                dashed_line,
            );
        }

        // Vertical time/tick line
        match self.basis {
            Basis::Time(_) => {
                let (rounded_timestamp, snap_ratio) =
                    self.snap_x_to_index(cursor_position.x, bounds, region);

                frame.stroke(
                    &Path::line(
                        Point::new(snap_ratio * bounds.width, 0.0),
                        Point::new(snap_ratio * bounds.width, bounds.height),
                    ),
                    dashed_line,
                );
                (rounded_price, rounded_timestamp)
            }
            Basis::Tick(aggregation) => {
                let (chart_x_min, chart_x_max) = (region.x, region.x + region.width);
                let x_range = chart_x_max - chart_x_min;
                let crosshair_pos = chart_x_min + (cursor_position.x / bounds.width) * x_range;

                let cell_index = (crosshair_pos / self.cell_width).round();

                let snapped_crosshair = cell_index * self.cell_width;
                let snap_ratio = if x_range.abs() > f32::EPSILON {
                    (snapped_crosshair - chart_x_min) / x_range
                } else {
                    0.5
                };

                let rounded_tick = (-cell_index as u64) * (u64::from(aggregation.0));

                if snap_ratio.is_finite() {
                    frame.stroke(
                        &Path::line(
                            Point::new(snap_ratio * bounds.width, 0.0),
                            Point::new(snap_ratio * bounds.width, bounds.height),
                        ),
                        dashed_line,
                    );
                }
                (rounded_price, rounded_tick)
            }
            Basis::Odb(_) => {
                let (chart_x_min, chart_x_max) = (region.x, region.x + region.width);
                let x_range = chart_x_max - chart_x_min;
                let crosshair_pos = chart_x_min + (cursor_position.x / bounds.width) * x_range;

                let cell_index = (crosshair_pos / self.cell_width).round();

                let snapped_crosshair = cell_index * self.cell_width;
                let snap_ratio = if x_range.abs() > f32::EPSILON {
                    (snapped_crosshair - chart_x_min) / x_range
                } else {
                    0.5
                };

                let rounded_index = if cell_index > 0.0 {
                    u64::MAX // sentinel: cursor is in forming bar territory
                } else {
                    (-cell_index) as u64
                };

                if snap_ratio.is_finite() {
                    frame.stroke(
                        &Path::line(
                            Point::new(snap_ratio * bounds.width, 0.0),
                            Point::new(snap_ratio * bounds.width, bounds.height),
                        ),
                        dashed_line,
                    );
                }
                (rounded_price, rounded_index)
            }
        }
    }

    pub(crate) fn draw_last_price_line(
        &self,
        frame: &mut canvas::Frame,
        palette: &Extended,
        region: Rectangle,
    ) {
        if let Some(price) = &self.last_price {
            let (last_price, line_color) = price.get_with_color(palette);
            let y_pos = self.price_to_y(last_price);

            let marker_line = Stroke::with_color(
                Stroke {
                    width: 1.0,
                    line_dash: LineDash {
                        segments: &[2.0, 2.0],
                        offset: 4,
                    },
                    ..Default::default()
                },
                line_color.scale_alpha(0.5),
            );

            frame.stroke(
                &Path::line(
                    Point::new(region.x, y_pos),
                    Point::new(region.x + region.width, y_pos),
                ),
                marker_line,
            );
        }
    }

    pub(crate) fn layout(&self) -> ViewConfig {
        let layout = &self.layout;
        ViewConfig {
            splits: layout.splits.clone(),
            autoscale: layout.autoscale,
            include_forming: layout.include_forming,
        }
    }

    pub(crate) fn y_labels_width(&self) -> Length {
        let precision = self.ticker_info.min_ticksize;

        let value = self.base_price_y.to_string(precision);
        let price_width = (value.len() as f32 * TEXT_SIZE * 0.8).max(72.0);

        // ODB timer label ("HH:MM:SS.mmm UTC") needs more room
        let width = if matches!(self.basis, Basis::Odb(_)) {
            price_width.max(135.0)
        } else {
            price_width
        };

        Length::Fixed(width.ceil())
    }

    pub(crate) fn snap_x_to_index(&self, x: f32, bounds: Size, region: Rectangle) -> (u64, f32) {
        let x_ratio = x / bounds.width;

        match self.basis {
            Basis::Time(timeframe) => {
                let interval = timeframe.to_milliseconds();
                let earliest = self.x_to_interval(region.x) as f64;
                let latest = self.x_to_interval(region.x + region.width) as f64;

                let millis_at_x = earliest + f64::from(x_ratio) * (latest - earliest);

                let rounded_timestamp = (millis_at_x / (interval as f64)).round() as u64 * interval;

                let snap_ratio = if latest - earliest > 0.0 {
                    ((rounded_timestamp as f64 - earliest) / (latest - earliest)) as f32
                } else {
                    0.5
                };

                (rounded_timestamp, snap_ratio)
            }
            Basis::Tick(aggregation) => {
                let (chart_x_min, chart_x_max) = (region.x, region.x + region.width);
                let chart_x = chart_x_min + x_ratio * (chart_x_max - chart_x_min);

                let cell_index = (chart_x / self.cell_width).round();
                let snapped_x = cell_index * self.cell_width;

                let snap_ratio = if chart_x_max - chart_x_min > 0.0 {
                    (snapped_x - chart_x_min) / (chart_x_max - chart_x_min)
                } else {
                    0.5
                };

                let rounded_tick = (-cell_index as u64) * u64::from(aggregation.0);

                (rounded_tick, snap_ratio)
            }
            Basis::Odb(_) => {
                let (chart_x_min, chart_x_max) = (region.x, region.x + region.width);
                let chart_x = chart_x_min + x_ratio * (chart_x_max - chart_x_min);

                let cell_index = (chart_x / self.cell_width).round();
                let snapped_x = cell_index * self.cell_width;

                let snap_ratio = if chart_x_max - chart_x_min > 0.0 {
                    (snapped_x - chart_x_min) / (chart_x_max - chart_x_min)
                } else {
                    0.5
                };

                let rounded_index = if cell_index > 0.0 {
                    u64::MAX // sentinel: cursor is in forming bar territory
                } else {
                    (-cell_index) as u64
                };

                (rounded_index, snap_ratio)
            }
        }
    }
}

use std::ops::RangeInclusive;

use iced::{Point, Size, Theme, widget::canvas};

use crate::chart::{
    ViewState,
    indicator::plot::{Plot, PlotTooltip, Series, TooltipFn, YScale},
};

#[derive(Clone, Copy)]
#[allow(unused)]
/// How to anchor bar heights.
pub enum Baseline {
    /// Use zero as baseline (classic volume). Extents: [0, max].
    Zero,
    /// Use the minimum value in the visible range. Extents: [min, max].
    Min,
    /// Use a fixed numeric baseline.
    Fixed(f32),
}

#[derive(Clone, Copy)]
/// What kind of bar to render.
// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97
pub enum BarClass {
    /// draw a single bar using secondary strong color
    Single,
    /// Stacked buy (success) + sell (danger) bar. Bottom = sell, top = buy.
    BuySell { buy: f32, sell: f32 },
    /// Solid bar colored by sign: positive = success, negative = danger.
    #[allow(dead_code)]
    Signed,
    /// Bar colored by candle direction: green if bullish (close >= open), red if bearish.
    /// Histogram direction shows +/-, color shows candle direction for divergence.
    CandleColored { bullish: bool },
}

pub struct BarPlot<V, CL, T> {
    /// Maps a datapoint to the scalar value represented by the bar (before baseline).
    pub value: V,
    pub bar_width_factor: f32,
    pub padding: f32,
    pub classify: CL, // Single, BuySell, or Signed
    pub tooltip: Option<TooltipFn<T>>,
    pub baseline: Baseline,
    _phantom: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
impl<V, CL, T> BarPlot<V, CL, T> {
    pub fn new(value: V, classify: CL) -> Self {
        Self {
            value,
            bar_width_factor: 0.9,
            padding: 0.0,
            classify,
            tooltip: None,
            baseline: Baseline::Zero,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn bar_width_factor(mut self, f: f32) -> Self {
        self.bar_width_factor = f;
        self
    }

    pub fn padding(mut self, p: f32) -> Self {
        self.padding = p;
        self
    }

    pub fn baseline(mut self, b: Baseline) -> Self {
        self.baseline = b;
        self
    }

    pub fn with_tooltip<F>(mut self, tooltip: F) -> Self
    where
        F: Fn(&T, Option<&T>) -> PlotTooltip + 'static,
    {
        self.tooltip = Some(Box::new(tooltip));
        self
    }
}

impl<S, V, CL> Plot<S> for BarPlot<V, CL, S::Y>
where
    S: Series,
    V: Fn(&S::Y) -> f32,
    CL: Fn(&S::Y) -> BarClass,
{
    fn y_extents(&self, datapoints: &S, range: RangeInclusive<u64>) -> Option<(f32, f32)> {
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        let mut n = 0u32;

        datapoints.for_each_in(range, |_, y| {
            let v = (self.value)(y);
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
            n += 1;
        });

        if n == 0 {
            return None;
        }

        let min_ext = match self.baseline {
            Baseline::Zero => min_v.min(0.0), // allow negative for bidirectional bars
            Baseline::Min => min_v,
            Baseline::Fixed(v) => v,
        };

        let max_ext = match self.baseline {
            Baseline::Zero => max_v.max(0.0), // always include zero line
            _ => max_v,
        };

        if min_ext >= max_ext && max_ext <= 0.0 && !matches!(self.baseline, Baseline::Zero) {
            return None;
        }

        let lowest = min_ext;
        let mut highest = max_ext.max(lowest + f32::EPSILON);
        if highest > lowest && self.padding > 0.0 {
            highest *= 1.0 + self.padding;
        }

        Some((lowest, highest))
    }

    fn draw(
        &self,
        frame: &mut canvas::Frame,
        ctx: &ViewState,
        theme: &Theme,
        datapoints: &S,
        range: RangeInclusive<u64>,
        scale: &YScale,
    ) {
        let palette = theme.extended_palette();
        let bar_width = ctx.cell_width * self.bar_width_factor;

        let baseline_value = match self.baseline {
            Baseline::Zero => 0.0,
            Baseline::Min => scale.min, // extents min
            Baseline::Fixed(v) => v,
        };
        let y_base = scale.to_y(baseline_value);

        datapoints.for_each_in(range, |x, y| {
            let center_x = ctx.interval_to_x(x);
            let left = center_x - (bar_width / 2.0);

            let total = (self.value)(y);
            let rel = total - baseline_value;

            let (top_y, h_total) = if rel > 0.0 {
                let y_total = scale.to_y(total);
                let h = (y_base - y_total).max(0.0);
                (y_total, h)
            } else if rel < 0.0 {
                // Negative bar: draw below baseline
                let y_total = scale.to_y(total);
                let h = (y_total - y_base).max(0.0);
                (y_base, h)
            } else {
                (y_base, 0.0)
            };
            if h_total <= 0.0 {
                return;
            }

            match (self.classify)(y) {
                BarClass::Single => {
                    frame.fill_rectangle(
                        Point::new(left, top_y),
                        Size::new(bar_width, h_total),
                        palette.secondary.strong.color,
                    );
                }
                BarClass::Signed => {
                    let color = if rel >= 0.0 {
                        palette.success.base.color
                    } else {
                        palette.danger.base.color
                    };
                    frame.fill_rectangle(
                        Point::new(left, top_y),
                        Size::new(bar_width, h_total),
                        color,
                    );
                }
                BarClass::CandleColored { bullish } => {
                    let color = if bullish {
                        palette.success.base.color
                    } else {
                        palette.danger.base.color
                    };
                    frame.fill_rectangle(
                        Point::new(left, top_y),
                        Size::new(bar_width, h_total),
                        color,
                    );
                }
                BarClass::BuySell { buy, sell } => {
                    // Stacked bar: sell (red) on bottom, buy (green) on top
                    let buy_color = palette.success.base.color;
                    let sell_color = palette.danger.base.color;

                    // Sell portion: from baseline up to sell height
                    if sell > 0.0 {
                        let y_sell_top = scale.to_y(sell);
                        let h_sell = (y_base - y_sell_top).max(0.0);
                        if h_sell > 0.0 {
                            frame.fill_rectangle(
                                Point::new(left, y_sell_top),
                                Size::new(bar_width, h_sell),
                                sell_color,
                            );
                        }
                    }

                    // Buy portion: stacked on top of sell
                    if buy > 0.0 {
                        let y_buy_top = scale.to_y(sell + buy);
                        let y_buy_bottom = scale.to_y(sell);
                        let h_buy = (y_buy_bottom - y_buy_top).max(0.0);
                        if h_buy > 0.0 {
                            frame.fill_rectangle(
                                Point::new(left, y_buy_top),
                                Size::new(bar_width, h_buy),
                                buy_color,
                            );
                        }
                    }
                }
            }
        });
    }

    fn tooltip_fn(&self) -> Option<&TooltipFn<S::Y>> {
        self.tooltip.as_ref()
    }
}

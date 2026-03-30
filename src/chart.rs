pub mod comparison;
pub mod heatmap;
pub mod indicator;
pub(crate) mod interaction;
pub mod keyboard_nav; // NOTE(fork): issue#100 — keyboard chart navigation
pub mod kline;
pub(crate) mod legend;
pub(crate) mod scale;
pub(crate) mod session;
pub(crate) mod view_state;

// Re-export all public types so downstream `use super::*` and
// `use crate::chart::{...}` continue to work unchanged.
pub(crate) use interaction::canvas_interaction;
pub use interaction::{AxisScaleClicked, Interaction, Message};
pub(crate) use interaction::{TEXT_SIZE, ZOOM_SENSITIVITY};
pub(crate) use legend::draw_volume_bar;
pub use legend::draw_watermark;
pub use view_state::{Caches, PlotConstants, ViewState};

use crate::connector::fetcher::{FetchRange, FetchSpec, RequestHandler};
use crate::style;
use crate::widget::multi_split::MultiSplit;
use crate::widget::tooltip;
use data::chart::{Autoscale, Basis, PlotData, indicator::Indicator};

use scale::{AxisLabelsX, AxisLabelsY};

use iced::widget::canvas::{self, Canvas};
use iced::{
    Alignment, Element, Length, Theme, Vector, padding,
    widget::{button, center, column, container, mouse_area, row, rule, text},
};

pub trait Chart: PlotConstants + canvas::Program<Message> {
    type IndicatorKind: Indicator;

    fn state(&self) -> &ViewState;

    fn mut_state(&mut self) -> &mut ViewState;

    fn invalidate_all(&mut self);

    fn invalidate_crosshair(&mut self);

    fn view_indicators(&'_ self, enabled: &[Self::IndicatorKind]) -> Vec<Element<'_, Message>>;

    fn visible_timerange(&self) -> Option<(u64, u64)>;

    fn interval_keys(&self) -> Option<Vec<u64>>;

    fn autoscaled_coords(&self) -> Vector;

    fn supports_fit_autoscaling(&self) -> bool;

    fn is_empty(&self) -> bool;
}

#[must_use = "Action must be handled by the caller"]
pub enum Action {
    ErrorOccurred(data::InternalError),
    RequestFetch(Vec<FetchSpec>),
}

pub fn update<T: Chart>(chart: &mut T, message: &Message) {
    match message {
        Message::DoubleClick(scale) => {
            let default_chart_width = T::default_cell_width(chart);
            let autoscaled_coords = chart.autoscaled_coords();
            let supports_fit_autoscaling = chart.supports_fit_autoscaling();

            let state = chart.mut_state();

            match scale {
                AxisScaleClicked::X => {
                    state.cell_width = default_chart_width;
                    state.translation = autoscaled_coords;
                }
                AxisScaleClicked::Y => {
                    if supports_fit_autoscaling {
                        state.layout.autoscale = Some(Autoscale::FitToVisible);
                        state.scaling = 1.0;
                    } else {
                        state.layout.autoscale = Some(Autoscale::CenterLatest);
                    }
                }
            }
        }
        Message::Translated(translation) => {
            let state = chart.mut_state();

            if let Some(Autoscale::FitToVisible) = state.layout.autoscale {
                state.translation.x = translation.x;
            } else {
                state.translation = *translation;
                state.layout.autoscale = None;
            }
        }
        Message::Scaled(scaling, translation) => {
            let state = chart.mut_state();
            state.scaling = *scaling;
            state.translation = *translation;

            state.layout.autoscale = None;
        }
        Message::AutoscaleToggled => {
            let supports_fit_autoscaling = chart.supports_fit_autoscaling();
            let state = chart.mut_state();

            let current_autoscale = state.layout.autoscale;
            state.layout.autoscale = {
                match current_autoscale {
                    None => Some(Autoscale::CenterLatest),
                    Some(Autoscale::CenterLatest) => {
                        if supports_fit_autoscaling {
                            Some(Autoscale::FitToVisible)
                        } else {
                            None
                        }
                    }
                    Some(Autoscale::FitToVisible) => None,
                }
            };

            if state.layout.autoscale.is_some() {
                state.scaling = 1.0;
            }
        }
        Message::XScaling(delta, cursor_to_center_x, is_wheel_scroll) => {
            let min_cell_width = T::min_cell_width(chart);
            let max_cell_width = T::max_cell_width(chart);

            let state = chart.mut_state();

            if !(*delta < 0.0 && state.cell_width > min_cell_width
                || *delta > 0.0 && state.cell_width < max_cell_width)
            {
                return;
            }

            let is_fit_to_visible_zoom =
                !is_wheel_scroll && matches!(state.layout.autoscale, Some(Autoscale::FitToVisible));

            let zoom_factor = if is_fit_to_visible_zoom {
                ZOOM_SENSITIVITY / 1.5
            } else if *is_wheel_scroll {
                ZOOM_SENSITIVITY
            } else {
                ZOOM_SENSITIVITY * 3.0
            };

            let new_width = (state.cell_width * (1.0 + delta / zoom_factor))
                .clamp(min_cell_width, max_cell_width);

            if is_fit_to_visible_zoom {
                let anchor_interval = {
                    let latest_x_coord = state.interval_to_x(state.latest_x);
                    if state.is_interval_x_visible(latest_x_coord) {
                        state.latest_x
                    } else {
                        let visible_region = state.visible_region(state.bounds.size());
                        state.x_to_interval(visible_region.x + visible_region.width)
                    }
                };

                let old_anchor_chart_x = state.interval_to_x(anchor_interval);

                state.cell_width = new_width;

                let new_anchor_chart_x = state.interval_to_x(anchor_interval);

                let shift = new_anchor_chart_x - old_anchor_chart_x;
                state.translation.x -= shift;
            } else {
                let (old_scaling, old_translation_x) = { (state.scaling, state.translation.x) };

                let latest_x = state.interval_to_x(state.latest_x);
                let is_interval_x_visible = state.is_interval_x_visible(latest_x);

                let cursor_chart_x = {
                    if *is_wheel_scroll || !is_interval_x_visible {
                        cursor_to_center_x / old_scaling - old_translation_x
                    } else {
                        latest_x / old_scaling - old_translation_x
                    }
                };

                let new_cursor_x = match state.basis {
                    Basis::Time(_) => {
                        let cursor_time = state.x_to_interval(cursor_chart_x);
                        state.cell_width = new_width;

                        state.interval_to_x(cursor_time)
                    }
                    Basis::Tick(_) | Basis::Odb(_) => {
                        let tick_index = cursor_chart_x / state.cell_width;
                        state.cell_width = new_width;

                        tick_index * state.cell_width
                    }
                };

                if *is_wheel_scroll || !is_interval_x_visible {
                    if !new_cursor_x.is_nan() && !cursor_chart_x.is_nan() {
                        state.translation.x -= new_cursor_x - cursor_chart_x;
                    }

                    state.layout.autoscale = None;
                }
            }
        }
        Message::YScaling(delta, cursor_to_center_y, is_wheel_scroll) => {
            let min_cell_height = T::min_cell_height(chart);
            let max_cell_height = T::max_cell_height(chart);

            let state = chart.mut_state();

            if state.layout.autoscale == Some(Autoscale::FitToVisible) {
                state.layout.autoscale = None;
            }

            if *delta < 0.0 && state.cell_height > min_cell_height
                || *delta > 0.0 && state.cell_height < max_cell_height
            {
                let (old_scaling, old_translation_y) = { (state.scaling, state.translation.y) };

                let zoom_factor = if *is_wheel_scroll {
                    ZOOM_SENSITIVITY
                } else {
                    ZOOM_SENSITIVITY * 3.0
                };

                let new_height = (state.cell_height * (1.0 + delta / zoom_factor))
                    .clamp(min_cell_height, max_cell_height);

                let cursor_chart_y = cursor_to_center_y / old_scaling - old_translation_y;

                let cursor_price = state.y_to_price(cursor_chart_y);

                state.cell_height = new_height;

                let new_cursor_y = state.price_to_y(cursor_price);

                state.translation.y -= new_cursor_y - cursor_chart_y;

                if *is_wheel_scroll {
                    state.layout.autoscale = None;
                }
            }
        }
        Message::BoundsChanged(bounds) => {
            let state = chart.mut_state();

            // calculate how center shifted
            let old_center_x = state.bounds.width / 2.0;
            let new_center_x = bounds.width / 2.0;
            let center_delta_x = (new_center_x - old_center_x) / state.scaling;

            state.bounds = *bounds;

            if state.layout.autoscale != Some(Autoscale::CenterLatest) {
                state.translation.x += center_delta_x;
            }
        }
        Message::SplitDragged(split, size) => {
            let state = chart.mut_state();

            if let Some(split) = state.layout.splits.get_mut(*split) {
                *split = (size * 100.0).round() / 100.0;
            }
        }
        Message::CrosshairMoved => {
            // Flush any pending zoom redraw on cursor move (trailing edge).
            let state = chart.mut_state();
            if state.zoom_pending_redraw {
                state.zoom_pending_redraw = false;
                state.last_zoom_invalidation = std::time::Instant::now();
                chart.invalidate_all();
            }
            return chart.invalidate_crosshair();
        }
    }

    // Throttle zoom invalidations to ~30fps. State (scaling, translation, cell_width)
    // is always updated above, but the expensive cache clear + redraw is skipped when
    // scroll events arrive faster than 33ms apart (trackpad momentum).
    let is_zoom_message = matches!(
        message,
        Message::Scaled(..) | Message::XScaling(..) | Message::YScaling(..)
    );
    if is_zoom_message {
        let state = chart.mut_state();
        let now = std::time::Instant::now();
        if now.duration_since(state.last_zoom_invalidation) >= std::time::Duration::from_millis(33)
        {
            state.last_zoom_invalidation = now;
            state.zoom_pending_redraw = false;
            chart.invalidate_all();
        } else {
            // Skip heavy redraw — just mark pending for trailing edge flush.
            state.zoom_pending_redraw = true;
            chart.invalidate_crosshair(); // lightweight: keep crosshair responsive
        }
    } else {
        chart.invalidate_all();
    }
}

pub fn view<'a, T: Chart>(
    chart: &'a T,
    indicators: &'a [T::IndicatorKind],
    timezone: data::UserTimezone,
) -> Element<'a, Message> {
    if chart.is_empty() {
        return center(text("Waiting for data...").size(16)).into();
    }

    let state = chart.state();
    // Propagate timezone into ViewState so canvas::Program::draw can read it.
    state.timezone.set(timezone);

    let axis_labels_x = Canvas::new(AxisLabelsX {
        labels_cache: &state.cache.x_labels,
        scaling: state.scaling,
        translation_x: state.translation.x,
        max: state.latest_x,
        basis: state.basis,
        cell_width: state.cell_width,
        timezone,
        chart_bounds: state.bounds,
        interval_keys: chart.interval_keys(),
        autoscaling: state.layout.autoscale,
    })
    .width(Length::Fill)
    .height(Length::Fill);

    let buttons = {
        let (autoscale_btn_placeholder, autoscale_btn_tooltip) = match state.layout.autoscale {
            Some(Autoscale::CenterLatest) => (text("C"), Some("Center last price")),
            Some(Autoscale::FitToVisible) => (text("A"), Some("Auto")),
            None => (text("C"), Some("Toggle autoscaling")),
        };
        let is_active = state.layout.autoscale.is_some();

        let autoscale_button = button(
            autoscale_btn_placeholder
                .size(10)
                .align_x(Alignment::Center)
                .align_y(Alignment::Center),
        )
        .height(Length::Fill)
        .on_press(Message::AutoscaleToggled)
        .style(move |theme: &Theme, status| style::button::transparent(theme, status, is_active));

        row![
            iced::widget::space::horizontal(),
            tooltip(
                autoscale_button,
                autoscale_btn_tooltip,
                iced::widget::tooltip::Position::Top
            ),
        ]
        .padding(2)
    };

    let y_labels_width = state.y_labels_width();

    let content = {
        let axis_labels_y = Canvas::new(AxisLabelsY {
            labels_cache: &state.cache.y_labels,
            translation_y: state.translation.y,
            scaling: state.scaling,
            decimals: state.decimals,
            min: state.base_price_y.to_f32_lossy(),
            last_price: state.last_price,
            last_trade_time: state.last_trade_time,
            tick_size: state.tick_size.to_f32_lossy(),
            cell_height: state.cell_height,
            basis: state.basis,
            chart_bounds: state.bounds,
        })
        .width(Length::Fill)
        .height(Length::Fill);

        let main_chart: Element<_> = row![
            container(Canvas::new(chart).width(Length::Fill).height(Length::Fill))
                .width(Length::FillPortion(10))
                .height(Length::FillPortion(120)),
            rule::vertical(1).style(style::split_ruler),
            container(
                mouse_area(axis_labels_y)
                    .on_double_click(Message::DoubleClick(AxisScaleClicked::Y))
            )
            .width(y_labels_width)
            .height(Length::FillPortion(120))
        ]
        .into();

        let indicators = chart.view_indicators(indicators);

        if indicators.is_empty() {
            main_chart
        } else {
            let panels = std::iter::once(main_chart)
                .chain(indicators)
                .collect::<Vec<_>>();

            // Safety net: clamp splits to exactly panels.len()-1 so a stale
            // saved state (more splits than current subplots) never panics.
            let n = panels.len() - 1;
            let splits = &state.layout.splits[..n.min(state.layout.splits.len())];

            MultiSplit::new(panels, splits, |index, position| {
                Message::SplitDragged(index, position)
            })
            .into()
        }
    };

    column![
        content,
        rule::horizontal(1).style(style::split_ruler),
        row![
            container(
                mouse_area(axis_labels_x)
                    .on_double_click(Message::DoubleClick(AxisScaleClicked::X))
            )
            .padding(padding::right(1))
            .width(Length::FillPortion(10))
            .height(Length::Fixed(26.0)),
            buttons.width(y_labels_width).height(Length::Fixed(26.0))
        ]
    ]
    .padding(padding::left(1).right(1).bottom(1))
    .into()
}

pub(crate) fn request_fetch(handler: &mut RequestHandler, range: FetchRange) -> Option<Action> {
    let range_clone = range.clone();
    match handler.add_request(range) {
        Ok(Some(req_id)) => {
            let fetch_spec = FetchSpec {
                req_id,
                fetch: range_clone,
                stream: None,
            };
            Some(Action::RequestFetch(vec![fetch_spec]))
        }
        Ok(None) => None,
        Err(reason) => {
            log::error!("Failed to request: {}", reason);
            // TODO: handle this more explicitly, maybe by returning Action::ErrorOccurred
            None
        }
    }
}

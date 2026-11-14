use crate::chart::Chart;
use data::chart::Autoscale;

use iced::widget::canvas::{self, Event};
use iced::{Point, Rectangle, Vector, keyboard, mouse};

use crate::widget::multi_split::DRAG_SIZE;

pub(crate) const ZOOM_SENSITIVITY: f32 = 60.0;

/// Multiplicative zoom rate (d3 convention): each scroll unit multiplies scale by 2^ZOOM_RATE.
/// 0.003 → ~139 ticks to double/halve. Perceptually uniform at every zoom level.
pub(crate) const ZOOM_RATE: f32 = 0.003;
pub(crate) const TEXT_SIZE: f32 = 12.0;

#[derive(Default, Debug, Clone, Copy)]
pub enum Interaction {
    #[default]
    None,
    Zoomin {
        last_position: Point,
    },
    Panning {
        translation: Vector,
        start: Point,
    },
    Ruler {
        start: Option<Point>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum AxisScaleClicked {
    X,
    Y,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    Translated(Vector),
    Scaled(f32, Vector),
    AutoscaleToggled,
    /// Toggle "freeze viewport" — when enabled, new bars do not push the
    /// visible window; translation auto-compensates by cell_width per
    /// appended bar so user-visible bars remain at stable screen positions.
    FreezeToggled,
    CrosshairMoved,
    YScaling(f32, f32, bool),
    XScaling(f32, f32, bool),
    BoundsChanged(Rectangle),
    SplitDragged(usize, f32),
    DoubleClick(AxisScaleClicked),
}

pub(crate) fn canvas_interaction<T: Chart>(
    chart: &T,
    interaction: &mut Interaction,
    event: &Event,
    bounds: Rectangle,
    cursor: mouse::Cursor,
) -> Option<canvas::Action<Message>> {
    if chart.state().bounds != bounds {
        return Some(canvas::Action::publish(Message::BoundsChanged(bounds)));
    }

    let shrunken_bounds = bounds.shrink(DRAG_SIZE * 4.0);
    let cursor_position = cursor.position_in(shrunken_bounds);

    if let Event::Mouse(mouse::Event::ButtonReleased(_)) = event {
        match interaction {
            Interaction::Panning { .. } | Interaction::Zoomin { .. } => {
                *interaction = Interaction::None;
            }
            _ => {}
        }
    }

    if let Interaction::Ruler { .. } = interaction
        && cursor_position.is_none()
    {
        *interaction = Interaction::None;
    }

    match event {
        Event::Mouse(mouse_event) => {
            let state = chart.state();

            match mouse_event {
                mouse::Event::ButtonPressed(button) => {
                    let cursor_in_bounds = cursor_position?;

                    if let mouse::Button::Left = button {
                        match interaction {
                            Interaction::None
                            | Interaction::Panning { .. }
                            | Interaction::Zoomin { .. } => {
                                *interaction = Interaction::Panning {
                                    translation: state.translation,
                                    start: cursor_in_bounds,
                                };
                            }
                            Interaction::Ruler { start } if start.is_none() => {
                                *interaction = Interaction::Ruler {
                                    start: Some(cursor_in_bounds),
                                };
                            }
                            Interaction::Ruler { .. } => {
                                *interaction = Interaction::None;
                            }
                        }
                    }
                    Some(canvas::Action::request_redraw().and_capture())
                }
                mouse::Event::CursorMoved { .. } => match *interaction {
                    Interaction::Panning { translation, start } => {
                        let cursor_in_bounds = cursor_position?;
                        let msg = Message::Translated(
                            translation + (cursor_in_bounds - start) * (1.0 / state.scaling),
                        );
                        Some(canvas::Action::publish(msg).and_capture())
                    }
                    Interaction::None | Interaction::Ruler { .. } => {
                        Some(canvas::Action::publish(Message::CrosshairMoved))
                    }
                    _ => None,
                },
                mouse::Event::WheelScrolled { delta } => {
                    cursor_position?;

                    let default_cell_width = T::default_cell_width(chart);
                    let min_cell_width = T::min_cell_width(chart);
                    let max_cell_width = T::max_cell_width(chart);
                    let max_scaling = T::max_scaling(chart);
                    let min_scaling = T::min_scaling(chart);

                    if matches!(interaction, Interaction::Panning { .. }) {
                        return Some(canvas::Action::capture());
                    }

                    let cursor_to_center = cursor.position_from(bounds.center())?;
                    // Lines (mouse wheel): ~1-3 per tick. Pixels (trackpad): ~10-50 per gesture.
                    // Scale pixel deltas down to match line-based sensitivity.
                    let y = match *delta {
                        mouse::ScrollDelta::Lines { y, .. } => y,
                        mouse::ScrollDelta::Pixels { y, .. } => y / 10.0,
                    };

                    if let Some(Autoscale::FitToVisible) = state.layout.autoscale {
                        // is_wheel_scroll=false keeps FitToVisible active (Y-axis auto-fit).
                        // Uses zoom_factor = ZOOM_SENSITIVITY / 1.5 = 40 — responsive enough
                        // after the pixel delta /10 normalization.
                        return Some(
                            canvas::Action::publish(Message::XScaling(
                                y,
                                cursor_to_center.x,
                                false,
                            ))
                            .and_capture(),
                        );
                    }

                    let should_adjust_cell_width = match (y.signum(), state.scaling) {
                        (-1.0, scaling)
                            if scaling == max_scaling && state.cell_width > default_cell_width =>
                        {
                            true
                        }
                        (1.0, scaling)
                            if scaling == min_scaling && state.cell_width < default_cell_width =>
                        {
                            true
                        }
                        (1.0, scaling)
                            if scaling == max_scaling && state.cell_width < max_cell_width =>
                        {
                            true
                        }
                        (-1.0, scaling)
                            if scaling == min_scaling && state.cell_width > min_cell_width =>
                        {
                            true
                        }
                        _ => false,
                    };

                    if should_adjust_cell_width {
                        return Some(
                            canvas::Action::publish(Message::XScaling(
                                y / 2.0,
                                cursor_to_center.x,
                                true,
                            ))
                            .and_capture(),
                        );
                    }

                    // normal scaling cases
                    if (y < 0.0 && state.scaling > min_scaling)
                        || (y > 0.0 && state.scaling < max_scaling)
                    {
                        let old_scaling = state.scaling;
                        // Multiplicative zoom (d3 formula): perceptually uniform at every level.
                        // Each tick multiplies by 2^(delta * ZOOM_RATE) — same % change always.
                        let scaling = (state.scaling * 2.0f32.powf(y * ZOOM_RATE))
                            .clamp(min_scaling, max_scaling);

                        let denominator = old_scaling * scaling;
                        let vector_diff = if denominator.abs() > 0.0001 {
                            let factor = scaling - old_scaling;
                            Vector::new(
                                cursor_to_center.x * factor / denominator,
                                cursor_to_center.y * factor / denominator,
                            )
                        } else {
                            Vector::default()
                        };

                        let translation = state.translation - vector_diff;

                        return Some(
                            canvas::Action::publish(Message::Scaled(scaling, translation))
                                .and_capture(),
                        );
                    }

                    Some(canvas::Action::capture())
                }
                _ => None,
            }
        }
        Event::Keyboard(keyboard_event) => {
            // NOTE(fork): cursor guard moved into Shift arm so arrow keys work
            // without the cursor being on the chart. See keyboard_nav.rs / issue#100.
            match keyboard_event {
                iced::keyboard::Event::KeyPressed { key, .. } => match key.as_ref() {
                    keyboard::Key::Named(keyboard::key::Named::Shift) => {
                        cursor_position?; // ruler requires cursor on-chart
                        *interaction = Interaction::Ruler { start: None };
                        Some(canvas::Action::request_redraw().and_capture())
                    }
                    keyboard::Key::Named(keyboard::key::Named::Escape) => {
                        *interaction = Interaction::None;
                        Some(canvas::Action::request_redraw().and_capture())
                    }
                    _ => super::keyboard_nav::process(keyboard_event, chart.state())
                        .map(|msg| canvas::Action::publish(msg).and_capture()),
                },
                _ => None,
            }
        }
        _ => None,
    }
}

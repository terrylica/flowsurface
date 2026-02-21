// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/100
//! Fork-specific keyboard navigation for chart panning.
//!
//! Wired into `canvas_interaction()` in `src/chart.rs` via the `_ =>` catchall
//! in the keyboard event block. All fork keyboard logic lives here; the upstream
//! file gets only a one-line delegation call.
//!
//! Sign convention (verified against mouse drag path):
//!   `interval_to_x` returns `-(idx * cell_width)` for Tick/RangeBar basis.
//!   Older bars → larger idx → more-negative cx. After the canvas transform
//!   `screen_x = center.x + scaling * (cx + translation.x)`, increasing
//!   `translation.x` shifts all content rightward, bringing older bars into view.
//!   So `ArrowLeft` (towards history) **increases** `translation.x`, mirroring
//!   a rightward mouse drag.

use super::Message;
use iced::{Vector, keyboard};
use iced::widget::canvas;

const BARS_SMALL: f32 = 10.0;
const BARS_LARGE: f32 = 50.0;

/// Handle a keyboard event and return a pan action, or `None` if the key is
/// not a navigation key.
///
/// Called from `canvas_interaction()` with the private `ViewState` fields
/// extracted at the call site (avoids exposing them to this submodule).
pub fn handle(
    event: &keyboard::Event,
    translation: Vector,
    cell_width: f32,
    scaling: f32,
    bounds_width: f32,
) -> Option<canvas::Action<Message>> {
    let keyboard::Event::KeyPressed { key, modifiers, .. } = event else {
        return None;
    };

    let shift = modifiers.shift();
    let bars = if shift { BARS_LARGE } else { BARS_SMALL };
    // Convert bar count → chart-coordinate step (pixels ÷ scaling)
    let step = bars * cell_width / scaling;

    let new_x = match key.as_ref() {
        // ArrowLeft  → older history (translation.x increases, chart slides right)
        keyboard::Key::Named(keyboard::key::Named::ArrowLeft) => translation.x + step,
        // ArrowRight → towards present (translation.x decreases, chart slides left)
        keyboard::Key::Named(keyboard::key::Named::ArrowRight) => translation.x - step,
        // PageUp → one full viewport width towards history
        keyboard::Key::Named(keyboard::key::Named::PageUp) => {
            translation.x + bounds_width / scaling
        }
        // PageDown → one full viewport width towards present
        keyboard::Key::Named(keyboard::key::Named::PageDown) => {
            translation.x - bounds_width / scaling
        }
        // Home → jump to latest bar (reset pan to centre-on-latest)
        keyboard::Key::Named(keyboard::key::Named::Home) => 0.0,
        _ => return None,
    };

    let new_translation = Vector::new(new_x, translation.y);
    Some(canvas::Action::publish(Message::Translated(new_translation)).and_capture())
}

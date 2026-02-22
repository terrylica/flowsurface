// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/100
//! Fork-specific keyboard navigation for chart panning and zooming.
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

use super::{Message, ViewState};
use iced::{Vector, keyboard};

const BARS_SMALL: f32 = 10.0;
const BARS_LARGE: f32 = 50.0;

/// Horizontal zoom delta per keypress (passed to `Message::XScaling`).
///
/// `is_wheel_scroll=false` → `zoom_factor = ZOOM_SENSITIVITY * 3.0 = 90.0` in chart.rs.
/// ZOOM_STEP / 90.0 ≈ 10% per normal press; ZOOM_STEP_LARGE / 90.0 ≈ 30% with Shift.
const ZOOM_STEP: f32 = 9.0;
const ZOOM_STEP_LARGE: f32 = 27.0;

/// Returns `true` if `key` is handled by [`process`], used by `keyboard::listen()`
/// in `main.rs` to forward only relevant keys to the focused chart pane.
///
/// Keeping the list here — next to the handlers — ensures a single source of truth:
/// adding a new key only requires editing this file.
pub fn is_nav_key(key: &keyboard::Key) -> bool {
    matches!(
        key,
        keyboard::Key::Named(
            keyboard::key::Named::ArrowLeft
                | keyboard::key::Named::ArrowRight
                | keyboard::key::Named::ArrowUp
                | keyboard::key::Named::ArrowDown
                | keyboard::key::Named::PageUp
                | keyboard::key::Named::PageDown
                | keyboard::key::Named::Home,
        )
    ) || matches!(
        key,
        keyboard::Key::Character(c) if matches!(
            c.as_ref(), "h" | "H" | "j" | "J" | "k" | "K" | "l" | "L" | "g"
        )
    )
}

/// Single entry point for keyboard navigation.
///
/// Returns a pan ([`Message::Translated`]) or horizontal-zoom ([`Message::XScaling`])
/// message, or `None` if the event is not a navigation key.
/// Callers do not need to know about the internal pan/zoom split.
pub fn process(event: &keyboard::Event, state: &ViewState) -> Option<Message> {
    pan(event, state).or_else(|| zoom(event))
}

/// Compute a pan `Message::Translated` from arrow/vim pan keys.
///
/// `←`/`h`   scroll 10 bars towards history
/// `→`/`l`   scroll 10 bars towards present
/// `H`/`L`   50-bar fast pan (Shift encodes fast variant for both forms)
/// `PageUp`  one full viewport width towards history
/// `PageDown` one full viewport width towards present
/// `Home`/`g` jump to latest bar
fn pan(event: &keyboard::Event, state: &ViewState) -> Option<Message> {
    let keyboard::Event::KeyPressed { key, modifiers, .. } = event else {
        return None;
    };

    // Shift applies to both Named arrow keys (via modifiers flag) and vim keys
    // (implicitly — pressing H means Shift is already held, so modifiers.shift()==true).
    // Either way, `step` is BARS_LARGE when Shift is held, BARS_SMALL otherwise.
    let bars = if modifiers.shift() { BARS_LARGE } else { BARS_SMALL };
    let step = bars * state.cell_width / state.scaling;

    let new_x = match key.as_ref() {
        // Named arrow keys
        keyboard::Key::Named(keyboard::key::Named::ArrowLeft) => state.translation.x + step,
        keyboard::Key::Named(keyboard::key::Named::ArrowRight) => state.translation.x - step,
        keyboard::Key::Named(keyboard::key::Named::PageUp) => {
            state.translation.x + state.bounds.width / state.scaling
        }
        keyboard::Key::Named(keyboard::key::Named::PageDown) => {
            state.translation.x - state.bounds.width / state.scaling
        }
        keyboard::Key::Named(keyboard::key::Named::Home) => 0.0,
        // vim hjkl — h/H and l/L collapse into the same arm because Shift is already
        // encoded in the character, so modifiers.shift() is true for H/L and step
        // already carries BARS_LARGE.
        keyboard::Key::Character(c) => match c {
            "h" | "H" => state.translation.x + step,
            "l" | "L" => state.translation.x - step,
            "g" => 0.0,
            _ => return None,
        },
        _ => return None,
    };

    Some(Message::Translated(Vector::new(new_x, state.translation.y)))
}

/// Compute a horizontal-zoom `Message::XScaling` from arrow/vim zoom keys.
///
/// `↑`/`k`   zoom in ~10% (widens bars)
/// `↓`/`j`   zoom out ~10% (narrows bars)
/// `K`/`J`   fast zoom ~30% (Shift variant)
///
/// Anchors on the latest visible bar when it is in view, otherwise on the viewport
/// centre — this matches the `is_wheel_scroll=false` path in `canvas_interaction()`.
fn zoom(event: &keyboard::Event) -> Option<Message> {
    let keyboard::Event::KeyPressed { key, modifiers, .. } = event else {
        return None;
    };

    let delta = match key.as_ref() {
        keyboard::Key::Named(keyboard::key::Named::ArrowUp) => {
            if modifiers.shift() { ZOOM_STEP_LARGE } else { ZOOM_STEP }
        }
        keyboard::Key::Named(keyboard::key::Named::ArrowDown) => {
            if modifiers.shift() { -ZOOM_STEP_LARGE } else { -ZOOM_STEP }
        }
        keyboard::Key::Character(c) => match c {
            "k" => ZOOM_STEP,
            "K" => ZOOM_STEP_LARGE,
            "j" => -ZOOM_STEP,
            "J" => -ZOOM_STEP_LARGE,
            _ => return None,
        },
        _ => return None,
    };

    // XScaling(delta, cursor_to_center_x=0.0, is_wheel_scroll=false)
    // is_wheel_scroll=false → zoom_factor = ZOOM_SENSITIVITY * 3.0 = 90.0
    // ZOOM_STEP=9.0 → 9/90 = 10% per keypress; ZOOM_STEP_LARGE=27.0 → 30%
    Some(Message::XScaling(delta, 0.0, false))
}

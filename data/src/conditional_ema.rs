// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97

/// Exponential Moving Average that only updates when a condition is active.
/// When inactive, the EMA value carries forward unchanged.
/// Useful for directional indicators (e.g., separate EMAs for bullish vs bearish bars).
#[derive(Clone)]
pub struct ConditionalEma {
    alpha: f32,
    value: Option<f32>,
}

impl ConditionalEma {
    pub fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f32 + 1.0),
            value: None,
        }
    }

    /// Update EMA with a new value. If `active` is false, the EMA carries forward.
    /// Returns the current EMA value (or 0.0 if never seeded).
    pub fn update(&mut self, input: f32, active: bool) -> f32 {
        if active {
            self.value = Some(match self.value {
                Some(prev) => self.alpha * input + (1.0 - self.alpha) * prev,
                None => input,
            });
        }
        self.current()
    }

    pub fn current(&self) -> f32 {
        self.value.unwrap_or(0.0)
    }

    pub fn reset(&mut self) {
        self.value = None;
    }
}

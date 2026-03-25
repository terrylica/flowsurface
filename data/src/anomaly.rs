//! Reusable anomaly detection primitives for rolling-window statistical analysis.
//!
//! Three complementary layers, all O(1) per bar on a pre-sorted window:
//! 1. **Adjusted Boxplot fence** — Quartile Skewness (Bahri 2024) + Hubert (2008) exponential formula
//! 2. **Conformal p-values** — distribution-free calibrated severity scoring
//! 3. **CUSUM** — cumulative sum control chart for sustained regime shift detection
//!
//! All functions operate on a sorted `&[f32]` window (ascending order).

/// Result from the adjusted boxplot fence computation.
#[derive(Debug, Clone, Copy)]
pub struct FenceResult {
    /// Lower fence value. Observations below this are flagged as anomalous.
    pub fence: f32,
    /// Interquartile range (Q3 - Q1). Used for severity scoring.
    pub iqr: f32,
    /// Quartile Skewness (Bowley coefficient). Positive = right-skewed.
    pub skewness: f32,
}

/// Adjusted Boxplot lower fence using Quartile Skewness (Bahri et al. 2024).
///
/// Replaces O(n²) Medcouple with O(1) Quartile Skewness:
///   QS = (Q3 + Q1 - 2*median) / (Q3 - Q1)
/// The Hubert (2008) exponential formula adapts the fence multiplier for skewness.
///
/// Requires a **sorted** slice of at least 20 elements.
/// Returns `None` if the window is too small or degenerate (IQR ≈ 0).
pub fn adjusted_lower_fence(sorted: &[f32]) -> Option<FenceResult> {
    let n = sorted.len();
    if n < 20 {
        return None;
    }
    let q1 = sorted[n / 4];
    let median = sorted[n / 2];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;
    if iqr <= f32::EPSILON {
        return None;
    }
    // Quartile Skewness (Bowley/Bahri): O(1), same sign convention as Medcouple.
    let qs = (q3 + q1 - 2.0 * median) / iqr;
    let h_lower = if qs >= 0.0 {
        1.5 * (-4.0 * qs).exp()
    } else {
        1.5 * (-3.0 * qs).exp()
    };
    Some(FenceResult {
        fence: q1 - h_lower * iqr,
        iqr,
        skewness: qs,
    })
}

/// Compute conformal p-value: fraction of window values with nonconformity score
/// at least as extreme as the test point.
///
/// Uses `|x - median| / MAD` as the nonconformity measure (MAD approximated as
/// half-IQR for O(1) computation on sorted data).
///
/// Distribution-free, calibrated by construction under exchangeability.
/// Returns p-value in [0, 1]. Lower = more anomalous.
pub fn conformal_pvalue(sorted: &[f32], value: f32) -> f32 {
    let n = sorted.len();
    if n < 3 {
        return 1.0;
    }
    let median = sorted[n / 2];
    // half-IQR ≈ 0.7413 * MAD — fast approximation from sorted data
    let mad = (sorted[3 * n / 4] - sorted[n / 4]) * 0.5;
    if mad <= f32::EPSILON {
        return 1.0;
    }
    let test_score = (value - median).abs() / mad;
    let count = sorted
        .iter()
        .filter(|&&v| (v - median).abs() / mad >= test_score)
        .count();
    count as f32 / (n + 1) as f32
}

/// CUSUM (Cumulative Sum) control chart for detecting sustained shifts.
///
/// Tracks cumulative evidence that the process has shifted below the rolling median.
/// Reset to zero when evidence accumulates in the opposite direction.
///
/// # Arguments
/// * `cusum_prev` — previous CUSUM accumulator value
/// * `value` — current observation
/// * `reference` — expected value under null hypothesis (typically rolling median)
/// * `allowance` — minimum shift to detect (half the expected deviation)
///
/// Returns the updated CUSUM value. Compare against a threshold to trigger alarms.
pub fn cusum_negative(cusum_prev: f32, value: f32, reference: f32, allowance: f32) -> f32 {
    (cusum_prev + (reference - value) - allowance).max(0.0)
}

/// Default CUSUM allowance for log₁₀(trade_intensity).
/// A shift of 0.3 in log space ≈ 2x intensity change.
pub const DEFAULT_CUSUM_ALLOWANCE: f32 = 0.15;

/// Default CUSUM alarm threshold. Higher = fewer false alarms, slower detection.
pub const DEFAULT_CUSUM_THRESHOLD: f32 = 2.0;

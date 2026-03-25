//! Reusable anomaly detection primitives for rolling-window statistical analysis.
//!
//! **Design for upstream extraction**: This module is pure math with zero framework
//! dependencies. Can be extracted into a standalone crate for ClickHouse columnar
//! features in opendeviationbar-py.
//!
//! Two complementary layers — **zero magic numbers**, only interpretable statistical parameters:
//! 1. **Conformal rank** — distribution-free anomaly detection via empirical rank (Vovk 2005)
//! 2. **CUSUM** — cumulative sum control chart for sustained regime shift detection (Page 1954)
//!
//! All functions operate on a sorted `&[f32]` window (ascending order).

/// Default significance level for anomaly detection.
/// α = 0.05 means "flag if in the bottom 5% of the rolling window."
/// This is a standard significance level, not a magic constant — it has a clear
/// probabilistic interpretation: P(false alarm) ≤ α under exchangeability.
pub const DEFAULT_ANOMALY_ALPHA: f32 = 0.05;

/// Conformal anomaly test: is `value` in the bottom α-fraction of `sorted`?
///
/// Pure rank-based — automatically handles skewness, heavy tails, multimodality.
/// No magic numbers, no skewness adjustments, no calibration constants.
///
/// # Guarantees
/// - Distribution-free: valid under exchangeability (no distributional assumptions)
/// - Finite-sample exact: P(false alarm) ≤ α for any n, any distribution
/// - Invariant to monotone transformations (rank-based)
///
/// # Returns
/// `true` if `value` is below the α-quantile of the window.
///
/// # Complexity
/// O(log n) — single binary search on the sorted window.
///
/// # References
/// - Vovk, Gammerman & Shafer (2005), "Algorithmic Learning in a Random World"
pub fn is_anomalous(sorted: &[f32], value: f32, alpha: f32) -> bool {
    let n = sorted.len();
    if n < 20 {
        return false;
    }
    let rank = sorted.partition_point(|&v| v < value);
    let p_value = (rank + 1) as f32 / (n + 1) as f32;
    p_value < alpha
}

/// Compute conformal p-value: empirical rank of `value` within the sorted window.
///
/// p-value = (rank + 1) / (n + 1), where rank = number of window values < value.
/// Lower p-value = more extreme (more anomalous).
///
/// # Guarantees
/// Same as `is_anomalous` — distribution-free, finite-sample exact.
///
/// # Returns
/// p-value in (0, 1]. Values near 0 are extreme outliers; 0.5 is median.
///
/// # Complexity
/// O(log n) — single binary search.
pub fn conformal_pvalue(sorted: &[f32], value: f32) -> f32 {
    let n = sorted.len();
    if n < 3 {
        return 1.0;
    }
    let rank = sorted.partition_point(|&v| v < value);
    (rank + 1) as f32 / (n + 1) as f32
}

/// CUSUM (Cumulative Sum) control chart for detecting sustained downward shifts.
///
/// Accumulates evidence that the process has shifted below the reference level.
/// Resets to zero when the process returns above reference + allowance.
///
/// # Arguments
/// * `cusum_prev` — previous CUSUM accumulator value (0.0 initially)
/// * `value` — current observation
/// * `reference` — expected value under null hypothesis (typically rolling median)
/// * `allowance` — minimum shift to detect (half the expected deviation under H₁)
///
/// # Returns
/// Updated CUSUM value (≥ 0). Compare against a threshold to trigger alarms.
///
/// # References
/// - Page (1954), "Continuous Inspection Schemes"
pub fn cusum_negative(cusum_prev: f32, value: f32, reference: f32, allowance: f32) -> f32 {
    (cusum_prev + (reference - value) - allowance).max(0.0)
}

/// Default CUSUM allowance for log₁₀(trade_intensity).
/// A shift of 0.3 in log space ≈ 2× intensity change; allowance = half that.
pub const DEFAULT_CUSUM_ALLOWANCE: f32 = 0.15;

/// Default CUSUM alarm threshold. Higher = fewer false alarms, slower detection.
pub const DEFAULT_CUSUM_THRESHOLD: f32 = 2.0;

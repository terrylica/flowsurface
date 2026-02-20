// GitHub Issue: https://github.com/terrylica/rangebar-py/issues/97
//! Rolling log-quantile percentile heatmap for trade intensity.
//!
//! Trade intensity has extreme power-law tails (BPR25: skewness=322, max/median = 700,000×).
//! This indicator computes a percentile rank of log10(intensity) within a rolling lookback
//! window and maps it to a thermal colour gradient (blue → cyan → green → amber → red).
//!
//! **Algorithm (no look-ahead bias)**:
//! ```text
//! log_val  = log10(intensity).max(0.0)
//! K_actual = clamp(round(cbrt(window.len())), 5, k_max)  ← adaptive bin count
//! rank     = count(window ≤ log_val) / window.len()       ← binary search on sorted window
//! bin      = ceil(rank × K_actual).clamp(1, K_actual)
//! t        = (bin − 1) / (K_actual − 1)                  ← normalise to [0, 1]
//! push log_val to ring-buffer; pop_front if len > lookback
//! ```
//!
//! The rank is computed BEFORE pushing the current bar — the current bar ranks
//! itself against previous bars only, giving zero look-ahead bias.

use crate::chart::{
    Caches, Message, ViewState,
    indicator::{
        indicator_row,
        kline::KlineIndicatorImpl,
        plot::{
            PlotTooltip,
            bar::{BarClass, BarPlot, Baseline},
        },
    },
};

use data::chart::{PlotData, kline::KlineDataPoint, kline::adaptive_k};
use exchange::{Kline, Trade};

use iced::widget::{center, text};
use std::collections::{BTreeMap, VecDeque};
use std::ops::RangeInclusive;

/// Per-bar data point for the heatmap indicator.
#[derive(Clone, Copy)]
struct HeatmapPoint {
    /// Raw trade intensity (t/s).
    intensity: f32,
    /// Bin index in 1..=k_actual.
    bin: u8,
    /// Adaptive K at the time this bar was processed.
    k_actual: u8,
}

impl HeatmapPoint {
    /// Normalise bin to [0, 1] for thermal colour lookup.
    fn t(self) -> f32 {
        if self.k_actual <= 1 {
            return 0.0;
        }
        (self.bin - 1) as f32 / (self.k_actual - 1) as f32
    }
}

/// Trade intensity heatmap indicator.
///
/// Distinct from [`TradeIntensityIndicator`] (raw single-colour bars) — this
/// variant applies rolling log-quantile percentile binning with a thermal
/// colour gradient for regime-adaptive intensity visualisation.
pub struct TradeIntensityHeatmapIndicator {
    cache: Caches,
    data: BTreeMap<u64, HeatmapPoint>,
    lookback: usize,
    k_max: u8,
}

impl TradeIntensityHeatmapIndicator {
    pub fn new() -> Self {
        Self::with_params(2000, 10)
    }

    pub fn with_params(lookback: usize, k_max: u8) -> Self {
        Self {
            cache: Caches::default(),
            data: BTreeMap::new(),
            lookback,
            k_max,
        }
    }

    /// Rebuild from all datapoints in the tick series.
    ///
    /// Maintains a ring buffer (insertion order) and a sorted frequency map
    /// (f32 bits → count) to compute O(log N) rank queries.
    fn compute_all(
        datapoints: impl Iterator<Item = (usize, f32)>,
        lookback: usize,
        k_max: u8,
    ) -> BTreeMap<u64, HeatmapPoint> {
        let mut ring: VecDeque<f32> = VecDeque::new();
        // Map from f32.to_bits() → frequency count.
        // Using bit representation preserves sort order for non-negative floats.
        let mut sorted: BTreeMap<u32, u32> = BTreeMap::new();
        let mut result = BTreeMap::new();

        for (idx, intensity) in datapoints {
            let log_val = intensity.log10().max(0.0);

            // --- Compute bin BEFORE pushing (no look-ahead) ---
            let n = ring.len();
            let k_actual = if n == 0 {
                5u8 // minimum K; first bar always gets bin 1
            } else {
                adaptive_k(n, k_max)
            };

            let bin = if n == 0 {
                1u8
            } else {
                let target_bits = log_val.to_bits();
                let rank_count: u32 = sorted
                    .range(..=target_bits)
                    .map(|(_, &c)| c)
                    .sum();
                let rank = rank_count as f32 / n as f32;
                ((rank * k_actual as f32).ceil() as u8).clamp(1, k_actual)
            };

            result.insert(idx as u64, HeatmapPoint { intensity, bin, k_actual });

            // --- Push current bar into window ---
            *sorted.entry(log_val.to_bits()).or_insert(0) += 1;
            ring.push_back(log_val);

            // Evict oldest if over capacity
            if ring.len() > lookback {
                let old = ring.pop_front().unwrap();
                let bits = old.to_bits();
                match sorted.get_mut(&bits) {
                    Some(cnt) if *cnt > 1 => *cnt -= 1,
                    _ => { sorted.remove(&bits); }
                }
            }
        }

        result
    }

    fn indicator_elem<'a>(
        &'a self,
        main_chart: &'a ViewState,
        visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        if self.data.is_empty() {
            return center(text("Intensity Heatmap: no microstructure data")).into();
        }

        let tooltip = |p: &HeatmapPoint, _next: Option<&HeatmapPoint>| {
            PlotTooltip::new(format!(
                "Intensity: {:.1} t/s (bin {}/{})",
                p.intensity, p.bin, p.k_actual
            ))
        };

        let bar_kind = |p: &HeatmapPoint| BarClass::Heatmap { t: p.t() };
        let value_fn = |p: &HeatmapPoint| p.intensity;

        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .baseline(Baseline::Zero)
            .with_tooltip(tooltip);

        indicator_row(main_chart, &self.cache, plot, &self.data, visible_range)
    }
}

impl KlineIndicatorImpl for TradeIntensityHeatmapIndicator {
    fn clear_all_caches(&mut self) {
        self.cache.clear_all();
    }

    fn clear_crosshair_caches(&mut self) {
        self.cache.clear_crosshair();
    }

    fn element<'a>(
        &'a self,
        chart: &'a ViewState,
        visible_range: RangeInclusive<u64>,
    ) -> iced::Element<'a, Message> {
        self.indicator_elem(chart, visible_range)
    }

    fn rebuild_from_source(&mut self, source: &PlotData<KlineDataPoint>) {
        match source {
            PlotData::TimeBased(_) => {
                self.data.clear();
            }
            PlotData::TickBased(tickseries) => {
                let iter = tickseries.datapoints.iter().enumerate().filter_map(|(idx, dp)| {
                    dp.microstructure.map(|m| (idx, m.trade_intensity))
                });
                self.data = Self::compute_all(iter, self.lookback, self.k_max);
            }
        }
        self.clear_all_caches();
    }

    fn on_insert_klines(&mut self, _klines: &[Kline]) {}

    fn on_insert_trades(
        &mut self,
        _trades: &[Trade],
        _old_dp_len: usize,
        source: &PlotData<KlineDataPoint>,
    ) {
        // Rolling window requires knowledge of the full ordered sequence.
        // Full rebuild is O(N) which is fast enough for range bars.
        self.rebuild_from_source(source);
    }

    fn on_ticksize_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn on_basis_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }
}

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
//! K_actual = max(round(cbrt(window.len())), 5)  ← fully adaptive, no cap
//! rank     = count(window ≤ log_val) / window.len()  ← binary search on sorted window
//! bin      = ceil(rank × K_actual).clamp(1, K_actual)
//! t        = (bin − 1) / (K_actual − 1)  ← normalise to [0, 1] for thermal colour
//! bar_height = bin  ← bounded 1..K, immune to outlier spikes
//! push log_val to ring-buffer; pop_front if len > lookback
//! ```
//!
//! The rank is computed BEFORE pushing the current bar — the current bar ranks
//! itself against previous bars only, giving zero look-ahead bias.
//!
//! **Incremental updates**: rolling window state (ring buffer + sorted freq map) is kept
//! on the struct. `on_insert_trades` processes only newly completed bars in O(1), avoiding
//! the O(N) full rebuild that caused UI freezes with large bar counts.

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
    // Rolling window state — maintained across calls for O(1) incremental updates.
    ring: VecDeque<f32>,
    /// Map from f32.to_bits() → frequency count for O(log N) rank queries.
    sorted: BTreeMap<u32, u32>,
    /// Number of tickseries datapoints processed so far (global index).
    next_idx: usize,
}

impl TradeIntensityHeatmapIndicator {
    pub fn new() -> Self {
        Self::with_lookback(2000)
    }

    pub fn with_lookback(lookback: usize) -> Self {
        Self {
            cache: Caches::default(),
            data: BTreeMap::new(),
            lookback,
            ring: VecDeque::new(),
            sorted: BTreeMap::new(),
            next_idx: 0,
        }
    }

    fn reset_state(&mut self) {
        self.ring.clear();
        self.sorted.clear();
        self.data.clear();
        self.next_idx = 0;
    }

    /// Process a single bar at `idx` with the given `intensity`.
    /// Updates the rolling window and stores the result in `self.data`.
    fn process_one(&mut self, idx: usize, intensity: f32) {
        let log_val = intensity.log10().max(0.0);

        // --- Compute bin BEFORE pushing (no look-ahead) ---
        let n = self.ring.len();
        let k_actual = if n == 0 {
            5u8 // minimum K; first bar always gets bin 1
        } else {
            adaptive_k(n)
        };

        let bin = if n == 0 {
            1u8
        } else {
            let target_bits = log_val.to_bits();
            let rank_count: u32 = self.sorted
                .range(..=target_bits)
                .map(|(_, &c)| c)
                .sum();
            let rank = rank_count as f32 / n as f32;
            ((rank * k_actual as f32).ceil() as u8).clamp(1, k_actual)
        };

        self.data.insert(idx as u64, HeatmapPoint { intensity, bin, k_actual });

        // --- Push current bar into window ---
        *self.sorted.entry(log_val.to_bits()).or_insert(0) += 1;
        self.ring.push_back(log_val);

        // Evict oldest if over capacity
        if self.ring.len() > self.lookback {
            let old = self.ring.pop_front().unwrap();
            let bits = old.to_bits();
            match self.sorted.get_mut(&bits) {
                Some(cnt) if *cnt > 1 => *cnt -= 1,
                _ => { self.sorted.remove(&bits); }
            }
        }
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
        // Height = bin level (1..K_actual) — bounded, outlier-immune.
        let value_fn = |p: &HeatmapPoint| p.bin as f32;

        // Pin Y scale to adaptive_k(lookback) so early K=5 bars don't appear
        // incorrectly short alongside mature K=13 bars when scrolling.
        let k_max = adaptive_k(self.lookback) as f32;
        let plot = BarPlot::new(value_fn, bar_kind)
            .bar_width_factor(0.9)
            .baseline(Baseline::Zero)
            .fixed_max(k_max)
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
        self.reset_state();
        if let PlotData::TickBased(tickseries) = source {
            for (idx, dp) in tickseries.datapoints.iter().enumerate() {
                if let Some(m) = dp.microstructure {
                    self.process_one(idx, m.trade_intensity);
                }
            }
            self.next_idx = tickseries.datapoints.len();
        }
        self.clear_all_caches();
    }

    fn on_insert_klines(&mut self, _klines: &[Kline]) {}

    fn on_insert_trades(
        &mut self,
        _trades: &[Trade],
        old_dp_len: usize,
        source: &PlotData<KlineDataPoint>,
    ) {
        match source {
            PlotData::TimeBased(_) => return,
            PlotData::TickBased(tickseries) => {
                let new_len = tickseries.datapoints.len();
                if self.next_idx == old_dp_len {
                    // Incremental path: only process newly completed bars.
                    for idx in old_dp_len..new_len {
                        if let Some(m) = tickseries.datapoints[idx].microstructure {
                            self.process_one(idx, m.trade_intensity);
                        }
                    }
                    self.next_idx = new_len;
                } else {
                    // State mismatch (e.g. after seek/reset): full rebuild.
                    self.rebuild_from_source(source);
                    return; // rebuild_from_source already cleared caches
                }
            }
        }
        self.clear_all_caches();
    }

    fn on_ticksize_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }

    fn on_basis_change(&mut self, source: &PlotData<KlineDataPoint>) {
        self.rebuild_from_source(source);
    }
}

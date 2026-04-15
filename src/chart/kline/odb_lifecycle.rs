// ODB lifecycle orchestration: watchdog, sentinel, viewport digest, telemetry.
// Extracted from kline/mod.rs to isolate fork-specific ODB complexity.
// GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/91
use super::*;

impl KlineChart {
    /// Trade feed liveness watchdog (dead-man's switch).
    /// Fires every frame via invalidate(). Alerts after 90s of no WS trades.
    pub(super) fn check_trade_feed_watchdog(&mut self) {
        if self.last_trade_received_ms > 0 {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let stale_ms = now_ms.saturating_sub(self.last_trade_received_ms);
            if stale_ms > 90_000 && !self.trade_feed_dead_alerted {
                self.trade_feed_dead_alerted = true;
                let stale_secs = stale_ms / 1000;
                log::error!(
                    "[watchdog] No WS trades for {stale_secs}s — feed may be dead. \
                     last_agg_id={:?}, ch_reconcile={}",
                    self.last_ws_agg_trade_id,
                    self.ch_reconcile_count,
                );
                if exchange::telegram::is_configured() {
                    let msg = format!(
                        "No WS trades for {stale_secs}s. last_agg_id={:?}, ch_reconcile={}",
                        self.last_ws_agg_trade_id, self.ch_reconcile_count,
                    );
                    tokio::spawn(async move {
                        exchange::telegram::alert(
                            exchange::telegram::Severity::Critical,
                            "trade-watchdog",
                            &msg,
                        )
                        .await;
                    });
                }
            }
        }
    }

    /// Sentinel: bar-level agg_trade_id continuity audit (every 60s, ODB only).
    /// Calls `self.audit_bar_continuity()` from odb_core.rs.
    pub(super) fn run_sentinel_audit(&mut self, now: Instant) {
        if !self.chart.basis.is_odb()
            || self.fetching_trades.0
            || now.duration_since(self.last_sentinel_audit) < std::time::Duration::from_secs(60)
        {
            return;
        }

        self.last_sentinel_audit = now;
        let anomalies = self.audit_bar_continuity();

        // Partition: healable gaps (re-fetch can fix) vs structural (day-boundary, overlaps)
        let healable: Vec<_> = anomalies
            .iter()
            .filter(|g| g.kind == BarGapKind::Gap)
            .collect();
        let day_boundary: Vec<_> = anomalies
            .iter()
            .filter(|g| g.kind == BarGapKind::DayBoundary)
            .collect();
        let overlaps: Vec<_> = anomalies
            .iter()
            .filter(|g| g.kind == BarGapKind::Overlap)
            .collect();

        let total_anomalies = anomalies.len();

        if total_anomalies > 0 && total_anomalies != self.sentinel_gap_count {
            self.sentinel_gap_count = total_anomalies;

            // Log all anomaly types
            if !healable.is_empty() {
                let missing: u64 = healable.iter().map(|g| g.missing_count).sum();
                log::warn!(
                    "[sentinel] {} healable gaps ({} missing agg_trade_ids)",
                    healable.len(),
                    missing,
                );
                for (i, gap) in healable.iter().take(3).enumerate() {
                    log::warn!(
                        "[sentinel]   gap {}: prev_last={} curr_first={} missing={}",
                        i + 1,
                        gap.prev_last_id,
                        gap.curr_first_id,
                        gap.missing_count,
                    );
                }
            }
            if !day_boundary.is_empty() {
                let missing: u64 = day_boundary.iter().map(|g| g.missing_count).sum();
                log::info!(
                    "[sentinel] {} day-boundary gaps ({} missing IDs, structural — kintsugi domain)",
                    day_boundary.len(),
                    missing,
                );
            }
            if !overlaps.is_empty() {
                let overlap_total: u64 = overlaps.iter().map(|g| g.missing_count).sum();
                log::warn!(
                    "[sentinel] {} overlapping bar pairs ({} shared agg_trade_ids)",
                    overlaps.len(),
                    overlap_total,
                );
                for (i, gap) in overlaps.iter().take(3).enumerate() {
                    log::warn!(
                        "[sentinel]   overlap {}: prev_last={} curr_first={} shared={}",
                        i + 1,
                        gap.prev_last_id,
                        gap.curr_first_id,
                        gap.missing_count,
                    );
                }
            }

            // Telegram: only alert for healable gaps or overlaps (not day-boundary)
            if exchange::telegram::is_configured() && (!healable.is_empty() || !overlaps.is_empty())
            {
                let mut detail = String::new();
                if !healable.is_empty() {
                    let missing: u64 = healable.iter().map(|g| g.missing_count).sum();
                    detail.push_str(&format!(
                        "{} healable gaps ({} missing IDs)\n",
                        healable.len(),
                        missing,
                    ));
                    for (i, gap) in healable.iter().take(5).enumerate() {
                        let secs = gap.bar_time_ms / 1000;
                        let nanos = ((gap.bar_time_ms % 1000) * 1_000_000) as u32;
                        let dt = chrono::DateTime::from_timestamp(secs as i64, nanos)
                            .map(|d| d.format("%Y-%m-%dT%H:%M UTC").to_string())
                            .unwrap_or_else(|| gap.bar_time_ms.to_string());
                        detail.push_str(&format!(
                            "\n{}. prev={} → curr={}\n   ({} missing, bar_time={})",
                            i + 1,
                            gap.prev_last_id,
                            gap.curr_first_id,
                            gap.missing_count,
                            dt,
                        ));
                    }
                }
                if !overlaps.is_empty() {
                    if !detail.is_empty() {
                        detail.push_str("\n\n");
                    }
                    let overlap_total: u64 = overlaps.iter().map(|g| g.missing_count).sum();
                    detail.push_str(&format!(
                        "{} overlapping bar pairs ({} shared IDs)",
                        overlaps.len(),
                        overlap_total,
                    ));
                }
                if !day_boundary.is_empty() {
                    detail.push_str(&format!(
                        "\n\n({} day-boundary gaps omitted — structural)",
                        day_boundary.len(),
                    ));
                }
                if !healable.is_empty() {
                    let now_ms = chrono::Utc::now().timestamp_millis() as u64;
                    let today_midnight_ms = (now_ms / 86_400_000) * 86_400_000;
                    let min_gap_time = healable
                        .iter()
                        .map(|g| g.prev_bar_time_ms)
                        .min()
                        .unwrap_or(0);
                    if min_gap_time >= today_midnight_ms {
                        detail
                            .push_str("\n\nLive-session gap — OdbCatchup handles (no CH refetch)");
                    } else {
                        detail.push_str("\n\nTriggering CH kline re-fetch...");
                    }
                }

                tokio::spawn(async move {
                    exchange::telegram::alert(
                        exchange::telegram::Severity::Warning,
                        "sentinel",
                        &detail,
                    )
                    .await;
                });
            }

            // Only trigger re-fetch for healable gaps (not day-boundary or overlaps)
            if !healable.is_empty() && !self.sentinel_refetch_pending {
                self.sentinel_refetch_pending = true;
                // Use prev_bar_time_ms (the OLDER side of the gap) to determine if the
                // gap has historical CH coverage. bar_time_ms (newer side) can be in
                // today's live session even when the gap itself spans historical data.
                self.sentinel_healable_gap_min_time_ms =
                    healable.iter().map(|g| g.prev_bar_time_ms).min();
                log::info!(
                    "[sentinel] triggering kline re-fetch to heal {} gaps",
                    healable.len()
                );
            }
        } else if total_anomalies == 0 && self.sentinel_gap_count > 0 {
            let prev_count = self.sentinel_gap_count;
            self.sentinel_gap_count = 0;
            self.sentinel_refetch_pending = false;
            self.sentinel_healable_gap_min_time_ms = None;

            log::info!("[sentinel] all {} previous anomalies healed", prev_count);

            if exchange::telegram::is_configured() {
                let msg = format!(
                    "All {} inter-bar anomalies healed (kintsugi repair confirmed)",
                    prev_count,
                );
                tokio::spawn(async move {
                    exchange::telegram::alert(
                        exchange::telegram::Severity::Recovery,
                        "sentinel",
                        &msg,
                    )
                    .await;
                });
            }
        }
    }

    /// Viewport digest: periodic bar quality summary (every 60s, ODB only).
    /// Always-on (not behind telemetry feature flag) so we can bootstrap
    /// analysis of what the user sees from the log file alone.
    pub(super) fn emit_viewport_digest(&mut self, now: Instant) {
        if !self.chart.basis.is_odb()
            || now.duration_since(self.last_viewport_digest) < std::time::Duration::from_secs(60)
        {
            return;
        }

        let PlotData::TickBased(tick_aggr) = &self.data_source else {
            return;
        };
        if tick_aggr.datapoints.is_empty() {
            return;
        }

        self.last_viewport_digest = now;
        let threshold_dbps = match self.chart.basis {
            Basis::Odb(d) => d,
            _ => 250,
        };
        let total = tick_aggr.datapoints.len();
        let newest_ts = tick_aggr
            .datapoints
            .last()
            .map(|dp| dp.kline.time)
            .unwrap_or(0);
        let oldest_ts = tick_aggr
            .datapoints
            .first()
            .map(|dp| dp.kline.time)
            .unwrap_or(0);
        let min_expected = (threshold_dbps as f32 / 10.0) * 0.8;

        // Scan last 50 bars for quality stats
        let tail_n = total.min(50);
        let tail = &tick_aggr.datapoints[total - tail_n..];
        let mut min_range: f32 = f32::MAX;
        let mut max_range: f32 = 0.0;
        let mut sum_range: f32 = 0.0;
        let mut suspect_count: usize = 0;
        for dp in tail {
            let k = &dp.kline;
            let range = ((k.high.to_f32() - k.low.to_f32()) / k.open.to_f32()) * 10_000.0;
            min_range = min_range.min(range);
            max_range = max_range.max(range);
            sum_range += range;
            if range < min_expected {
                suspect_count += 1;
            }
        }
        let avg_range = sum_range / tail_n as f32;

        let forming_info = self.odb_processor.as_ref().and_then(|p| {
            p.get_incomplete_bar().map(|b| {
                let dev = ((b.high.to_f64() - b.low.to_f64()) / b.open.to_f64()) * 10_000.0;
                (b.agg_record_count, dev)
            })
        });

        log::warn!(
            "[viewport] BPR{} total={} oldest={} newest={} \
             tail{}=[min={:.1} avg={:.1} max={:.1} dbps] suspect={} \
             forming=[{}] reconciled={}",
            threshold_dbps / 10,
            total,
            oldest_ts,
            newest_ts,
            tail_n,
            min_range,
            avg_range,
            max_range,
            suspect_count,
            forming_info.map_or("none".to_string(), |(tc, dev)| {
                format!("{tc} trades, {dev:.1}dbps")
            }),
            self.ch_reconcile_count,
        );

        if suspect_count > tail_n / 4 {
            log::warn!(
                "[viewport] >25% of recent bars are under-threshold ({}/{} < {:.0} dbps). \
                 Upstream pipeline may be producing malformed bars.",
                suspect_count,
                tail_n,
                min_expected,
            );
        }
    }

    /// Telemetry snapshot: emit ChartSnapshot (every 30s, ODB only).
    #[cfg(feature = "telemetry")]
    pub(super) fn emit_telemetry_snapshot(&mut self, now: Instant) {
        if !self.chart.basis.is_odb()
            || now.duration_since(self.last_snapshot) < std::time::Duration::from_secs(30)
        {
            return;
        }

        self.last_snapshot = now;
        use data::telemetry::{self, TelemetryEvent};
        if let PlotData::TickBased(ref tick_aggr) = self.data_source {
            let telem_dbps = if let data::chart::Basis::Odb(d) = self.chart.basis {
                d
            } else {
                0
            };
            let forming_ts = self
                .odb_processor
                .as_ref()
                .and_then(|p| p.get_incomplete_bar())
                .map(|b| (b.close_time / 1000) as u64); // us -> ms
            telemetry::emit(TelemetryEvent::ChartSnapshot {
                ts_ms: telemetry::now_ms(),
                symbol: self.chart.ticker_info.ticker.to_string(),
                threshold_dbps: telem_dbps,
                total_bars: tick_aggr.datapoints.len(),
                visible_bars: if self.chart.cell_width > 0.0 {
                    (self.chart.bounds.width / self.chart.cell_width).ceil() as usize
                } else {
                    0
                },
                newest_bar_ts: tick_aggr
                    .datapoints
                    .last()
                    .map(|dp| dp.kline.time)
                    .unwrap_or(0),
                oldest_bar_ts: tick_aggr
                    .datapoints
                    .first()
                    .map(|dp| dp.kline.time)
                    .unwrap_or(0),
                forming_bar_ts: forming_ts,
                rbp_completed_count: self.odb_completed_count,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use data::chart::Basis;

    #[test]
    fn cooldown_arithmetic_blocks_within_30s() {
        let last: u64 = 1_000_000;
        let now: u64 = 1_020_000; // 20s later
        assert!(now.saturating_sub(last) <= 30_000);
    }

    #[test]
    fn cooldown_arithmetic_allows_after_30s() {
        let last: u64 = 1_000_000;
        let now: u64 = 1_031_000; // 31s later
        assert!(now.saturating_sub(last) > 30_000);
    }

    #[test]
    fn cooldown_arithmetic_exact_boundary() {
        let last: u64 = 1_000_000;
        let now: u64 = 1_030_000; // exactly 30s
        assert!(now.saturating_sub(last) <= 30_000);
    }

    /// Guard logic: `!fetching_trades && !gap_fill_requested && basis.is_odb()
    ///               && now_ms.saturating_sub(last_trigger) > 30_000`
    fn guard_allows(
        fetching_trades: bool,
        gap_fill_requested: bool,
        basis: &Basis,
        now_ms: u64,
        last_trigger: u64,
    ) -> bool {
        !fetching_trades
            && !gap_fill_requested
            && basis.is_odb()
            && now_ms.saturating_sub(last_trigger) > 30_000
    }

    #[test]
    fn guard_composition_all_false() {
        // All guard conditions are "clear" -> trigger allowed
        assert!(guard_allows(
            false,
            false,
            &Basis::Odb(250),
            100_000,
            60_000
        ));
    }

    #[test]
    fn guard_fetching_blocks() {
        assert!(!guard_allows(
            true,
            false,
            &Basis::Odb(250),
            100_000,
            60_000
        ));
    }

    #[test]
    fn guard_already_requested_blocks() {
        assert!(!guard_allows(
            false,
            true,
            &Basis::Odb(250),
            100_000,
            60_000
        ));
    }

    #[test]
    fn guard_cooldown_blocks() {
        // last_trigger 20s ago -> within 30s cooldown
        assert!(!guard_allows(
            false,
            false,
            &Basis::Odb(250),
            100_000,
            80_000
        ));
    }

    #[test]
    fn guard_non_odb_blocks() {
        assert!(!guard_allows(
            false,
            false,
            &Basis::Time(exchange::Timeframe::M1),
            100_000,
            60_000
        ));
    }

    #[test]
    fn guard_all_clear_allows() {
        // All guards pass: not fetching, not requested, ODB basis, cooldown expired
        assert!(guard_allows(
            false,
            false,
            &Basis::Odb(500),
            200_000,
            100_000
        ));
    }
}

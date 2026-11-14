// Data insertion and aggregation query methods extracted from kline/mod.rs.
// These methods operate on KlineChart's data_source, indicators, and
// request_handler -- no canvas rendering or RefCell interaction.
use super::*;

impl KlineChart {
    pub fn insert_hist_klines(&mut self, req_id: uuid::Uuid, klines_raw: &[Kline]) {
        match self.data_source {
            PlotData::TimeBased(ref mut timeseries) => {
                timeseries.insert_klines(klines_raw);
                timeseries.insert_trades_existing_buckets(&self.raw_trades);

                self.indicators
                    .values_mut()
                    .filter_map(Option::as_mut)
                    .for_each(|indi| indi.on_insert_klines(klines_raw));

                if klines_raw.is_empty() {
                    self.request_handler
                        .mark_failed(req_id, "No data received".to_string());
                } else {
                    self.request_handler.mark_completed(req_id);
                }
                let _ = self.invalidate(None);
            }
            PlotData::TickBased(_) => {}
        }
    }

    pub fn insert_open_interest(&mut self, req_id: Option<uuid::Uuid>, oi_data: &[OIData]) {
        if let Some(req_id) = req_id {
            if oi_data.is_empty() {
                self.request_handler
                    .mark_failed(req_id, "No data received".to_string());
            } else {
                self.request_handler.mark_completed(req_id);
            }
        }

        if let Some(indi) = self.indicators[KlineIndicator::OpenInterest].as_mut() {
            indi.on_open_interest(oi_data);
        }
    }

    pub fn toggle_indicator(&mut self, indicator: KlineIndicator) {
        // Count only panel indicators (TradeIntensityHeatmap colours candles, not a panel).
        let prev_panel_count = self
            .indicators
            .iter()
            .filter(|(k, v)| v.is_some() && k.has_subplot())
            .count();

        if self.indicators[indicator].is_some() {
            self.indicators[indicator] = None;
        } else {
            let mut box_indi = indicator::kline::make_indicator(indicator, &self.kline_config);
            box_indi.rebuild_from_source(&self.data_source);
            self.indicators[indicator] = Some(box_indi);
        }

        if let Some(main_split) = self.chart.layout.splits.first() {
            let current_panel_count = self
                .indicators
                .iter()
                .filter(|(k, v)| v.is_some() && k.has_subplot())
                .count();
            self.chart.layout.splits = data::util::calc_panel_splits(
                *main_split,
                current_panel_count,
                Some(prev_panel_count),
            );
        }
    }

    pub(super) fn missing_data_task(&mut self) -> Option<Action> {
        // Sentinel refetch: clear existing bars so the fresh CH fetch fully replaces
        // the display (not just prepends). Without clearing, prepend_klines skips all
        // bars that are newer than the current oldest — which is all of them since we
        // fetch the N most recent bars. Clearing forces a full reload.
        //
        // Guard: live-session gaps (bars built after UTC midnight) are not yet committed
        // to CH. Clearing datapoints for them would wipe all live bars with no CH
        // replacement. Detect this by comparing the gap's bar_time against today's UTC
        // midnight; skip the destructive clear and let OdbCatchup (already triggered by
        // insert_trades_inner gap detection) handle intra-session gaps instead.
        if self.sentinel_refetch_pending && self.chart.basis.is_odb() {
            self.sentinel_refetch_pending = false;
            let now_ms = chrono::Utc::now().timestamp_millis() as u64;
            let today_midnight_ms = (now_ms / 86_400_000) * 86_400_000;
            let gap_is_live_session = self
                .sentinel_healable_gap_min_time_ms
                .map(|t| t >= today_midnight_ms)
                .unwrap_or(false);
            self.sentinel_healable_gap_min_time_ms = None;

            if gap_is_live_session {
                // Recency guard: gaps within the last 24h are likely live-session gaps
                // that OdbCatchup should handle — valid in both aion and legacy modes.
                log::warn!(
                    "[sentinel] live-session gap (post-midnight) — skipping CH refetch to \
                     avoid wiping live bars; OdbCatchup handles this"
                );
                return None;
            }

            if let PlotData::TickBased(tick_aggr) = &mut self.data_source {
                tick_aggr.datapoints.clear();
            }
            self.request_handler = RequestHandler::default();
            // u64::MAX signals "full reload — no time constraint" to build_odb_sql,
            // which uses the adaptive limit (20K/13K) instead of LIMIT 2000.
            let range = FetchRange::Kline(0, u64::MAX);
            return request_fetch(&mut self.request_handler, range);
        }

        match &self.data_source {
            PlotData::TimeBased(timeseries) => {
                let timeframe_ms = timeseries.interval.to_milliseconds();

                if timeseries.datapoints.is_empty() {
                    let latest = chrono::Utc::now().timestamp_millis() as u64;
                    let earliest = latest.saturating_sub(450 * timeframe_ms);

                    let range = FetchRange::Kline(earliest, latest);
                    if let Some(action) = request_fetch(&mut self.request_handler, range) {
                        return Some(action);
                    }
                }

                let (visible_earliest, visible_latest) = self.visible_timerange()?;
                let (kline_earliest, kline_latest) = timeseries.timerange();
                let earliest = visible_earliest.saturating_sub(visible_latest - visible_earliest);

                // priority 1, basic kline data fetch
                if visible_earliest < kline_earliest {
                    let range = FetchRange::Kline(earliest, kline_earliest);

                    if let Some(action) = request_fetch(&mut self.request_handler, range) {
                        return Some(action);
                    }
                }

                // priority 2, trades fetch
                if !self.fetching_trades.0
                    && is_trade_fetch_enabled()
                    && let Some((fetch_from, fetch_to)) =
                        timeseries.suggest_trade_fetch_range(visible_earliest, visible_latest)
                {
                    let range = FetchRange::Trades(fetch_from, fetch_to);
                    if let Some(action) = request_fetch(&mut self.request_handler, range) {
                        self.fetching_trades = (true, None);
                        return Some(action);
                    }
                }

                // priority 3, Open Interest data
                let ctx = indicator::kline::FetchCtx {
                    main_chart: &self.chart,
                    timeframe: timeseries.interval,
                    visible_earliest,
                    kline_latest,
                    prefetch_earliest: earliest,
                };
                for indi in self.indicators.values_mut().filter_map(Option::as_mut) {
                    if let Some(range) = indi.fetch_range(&ctx)
                        && let Some(action) = request_fetch(&mut self.request_handler, range)
                    {
                        return Some(action);
                    }
                }

                // priority 4, missing klines & integrity check
                if let Some(missing_keys) =
                    timeseries.check_kline_integrity(kline_earliest, kline_latest)
                {
                    let latest =
                        missing_keys.iter().max().unwrap_or(&visible_latest) + timeframe_ms;
                    let earliest =
                        missing_keys.iter().min().unwrap_or(&visible_earliest) - timeframe_ms;

                    let range = FetchRange::Kline(earliest, latest);
                    if let Some(action) = request_fetch(&mut self.request_handler, range) {
                        return Some(action);
                    }
                }
            }
            PlotData::TickBased(tick_aggr) => {
                if self.chart.basis.is_odb() {
                    if tick_aggr.datapoints.is_empty() {
                        // Initial fetch — u64::MAX signals "no time constraint, use adaptive
                        // limit" in build_odb_sql (20K for BPR25, 13K floor for others).
                        let range = FetchRange::Kline(0, u64::MAX);
                        return request_fetch(&mut self.request_handler, range);
                    }

                    // Request older data when scrolling left.
                    // TickAggr stores oldest-first; render iterates .rev().enumerate()
                    // so index 0 = newest (rightmost), index N-1 = oldest (leftmost).
                    let oldest_ts = tick_aggr.datapoints.first().unwrap().kline.time;

                    let visible_region = self.chart.visible_region(self.chart.bounds.size());
                    let (_earliest_idx, latest_idx) = self.chart.interval_range(&visible_region);
                    let total_bars = tick_aggr.datapoints.len() as u64;

                    // latest_idx is the left edge (oldest visible bar index).
                    // Fetch when it reaches 80% of loaded bars for smooth scrolling.
                    let fetch_threshold = total_bars.saturating_sub(total_bars / 5);
                    if latest_idx >= fetch_threshold {
                        let range = FetchRange::Kline(0, oldest_ts);
                        return request_fetch(&mut self.request_handler, range);
                    }
                }
            }
        }

        None
    }

    pub(super) fn calc_qty_scales(
        &self,
        earliest: u64,
        latest: u64,
        highest: Price,
        lowest: Price,
        step: PriceStep,
        cluster_kind: ClusterKind,
    ) -> f32 {
        let rounded_highest = highest.round_to_side_step(false, step).add_steps(1, step);
        let rounded_lowest = lowest.round_to_side_step(true, step).add_steps(-1, step);

        match &self.data_source {
            PlotData::TimeBased(timeseries) => timeseries
                .max_qty_ts_range(
                    cluster_kind,
                    earliest,
                    latest,
                    rounded_highest,
                    rounded_lowest,
                )
                .into(),
            PlotData::TickBased(tick_aggr) => {
                let earliest = earliest as usize;
                let latest = latest as usize;

                tick_aggr
                    .max_qty_idx_range(
                        cluster_kind,
                        earliest,
                        latest,
                        rounded_highest,
                        rounded_lowest,
                    )
                    .into()
            }
        }
    }
}

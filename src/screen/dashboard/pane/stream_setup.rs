// FILE-SIZE-OK: stream wiring + data insertion helpers -- single extraction unit from pane/mod.rs
// GitHub Issue: https://github.com/flowsurface-rs/flowsurface/pull/89

use super::{Content, Effect};
use crate::{
    chart::{self, comparison::ComparisonChart, kline::KlineChart},
    connector::ResolvedStream,
    screen::dashboard::panel::{ladder::Ladder, timeandsales::TimeAndSales},
};
use data::{
    chart::Basis,
    layout::pane::{ContentKind, PaneSetup, Settings},
};
use exchange::{
    Kline, OpenInterest, TickMultiplier, TickerInfo, Timeframe,
    adapter::{StreamKind, StreamTicksize},
};

/// Builds content and streams for a pane given tickers and content kind.
///
/// This is the body of `State::set_content_and_streams()`. Takes individual
/// field references to avoid borrow-checker conflicts with callers that hold
/// partial borrows on `State`.
pub(super) fn build_content_and_streams(
    content: &mut Content,
    streams: &mut ResolvedStream,
    settings: &mut Settings,
    tickers: Vec<TickerInfo>,
    kind: ContentKind,
) -> Vec<StreamKind> {
    if !(content.kind() == kind) {
        settings.selected_basis = None;
        settings.tick_multiply = None;
    }

    let base_ticker = tickers[0];
    let prev_base_ticker = streams.find_ready_map(|stream| match stream {
        StreamKind::Kline { ticker_info, .. }
        | StreamKind::OdbKline { ticker_info, .. }
        | StreamKind::Depth { ticker_info, .. }
        | StreamKind::Trades { ticker_info, .. } => Some(*ticker_info),
    });

    let derived_plan = PaneSetup::new(
        kind,
        base_ticker,
        prev_base_ticker,
        settings.selected_basis,
        settings.tick_multiply,
    );

    settings.selected_basis = derived_plan.basis;
    settings.tick_multiply = derived_plan.tick_multiplier;

    let (new_content, new_streams) = {
        let kline_stream = |ti: TickerInfo, tf: Timeframe| StreamKind::Kline {
            ticker_info: ti,
            timeframe: tf,
        };
        let depth_stream = |derived_plan: &PaneSetup| StreamKind::Depth {
            ticker_info: derived_plan.ticker_info,
            depth_aggr: derived_plan.depth_aggr,
            push_freq: derived_plan.push_freq,
        };
        let trades_stream = |derived_plan: &PaneSetup| StreamKind::Trades {
            ticker_info: derived_plan.ticker_info,
        };

        match kind {
            ContentKind::HeatmapChart => {
                let c = Content::new_heatmap(
                    content,
                    derived_plan.ticker_info,
                    settings,
                    derived_plan.price_step.to_f32_lossy(),
                );

                let s = vec![depth_stream(&derived_plan), trades_stream(&derived_plan)];

                (c, s)
            }
            ContentKind::FootprintChart => {
                let c = Content::new_kline(
                    kind,
                    content,
                    derived_plan.ticker_info,
                    settings,
                    derived_plan.price_step.to_f32_lossy(),
                );

                let s = by_basis_default(
                    derived_plan.basis,
                    Timeframe::M5,
                    |tf| {
                        vec![
                            trades_stream(&derived_plan),
                            kline_stream(derived_plan.ticker_info, tf),
                        ]
                    },
                    || vec![trades_stream(&derived_plan)],
                    |threshold| {
                        vec![
                            depth_stream(&derived_plan),
                            trades_stream(&derived_plan),
                            StreamKind::OdbKline {
                                ticker_info: derived_plan.ticker_info,
                                threshold_dbps: threshold,
                            },
                        ]
                    },
                );

                (c, s)
            }
            ContentKind::CandlestickChart => {
                let c = {
                    let base_ticker = tickers[0];
                    Content::new_kline(
                        kind,
                        content,
                        derived_plan.ticker_info,
                        settings,
                        base_ticker.min_ticksize.into(),
                    )
                };

                let s = by_basis_default(
                    derived_plan.basis,
                    Timeframe::M15,
                    |tf| vec![kline_stream(derived_plan.ticker_info, tf)],
                    || {
                        let depth_aggr = derived_plan
                            .ticker_info
                            .exchange()
                            .stream_ticksize(None, TickMultiplier(50));
                        let temp = PaneSetup {
                            depth_aggr,
                            ..derived_plan
                        };
                        vec![trades_stream(&temp)]
                    },
                    |threshold| {
                        vec![
                            StreamKind::OdbKline {
                                ticker_info: derived_plan.ticker_info,
                                threshold_dbps: threshold,
                            },
                            depth_stream(&derived_plan),
                            trades_stream(&derived_plan),
                        ]
                    },
                );

                (c, s)
            }
            // GitHub Issue: https://github.com/terrylica/opendeviationbar-py/issues/91
            ContentKind::OdbChart => {
                let c = Content::new_kline(
                    kind,
                    content,
                    derived_plan.ticker_info,
                    settings,
                    base_ticker.min_ticksize.into(),
                );
                let default_threshold = if derived_plan
                    .ticker_info
                    .ticker
                    .exchange
                    .venue()
                    == exchange::adapter::Venue::ClickHouse
                {
                    // 25 dbps: lowest overshoot for forex (5 dbps has 5-10x overshoot)
                    *data::chart::ODB_THRESHOLDS_FOREX.last().unwrap()
                } else {
                    250
                };
                let is_ch = derived_plan
                    .ticker_info
                    .ticker
                    .exchange
                    .venue()
                    == exchange::adapter::Venue::ClickHouse;
                let threshold = match derived_plan.basis {
                    Some(Basis::Odb(t)) if !is_ch => t,
                    Some(Basis::Odb(t))
                        if data::chart::ODB_THRESHOLDS_FOREX
                            .contains(&t) =>
                    {
                        t
                    }
                    _ => default_threshold,
                };
                // Sync selected_basis with the validated threshold
                settings.selected_basis = Some(Basis::Odb(threshold));
                let mut s = vec![StreamKind::OdbKline {
                    ticker_info: derived_plan.ticker_info,
                    threshold_dbps: threshold,
                }];
                // NOTE(fork): ClickHouse-only symbols (forex) have no
                // WebSocket streams — skip Trades/Depth.
                if derived_plan
                    .ticker_info
                    .ticker
                    .exchange
                    .venue()
                    != exchange::adapter::Venue::ClickHouse
                {
                    s.push(depth_stream(&derived_plan));
                    s.push(trades_stream(&derived_plan));
                }
                (c, s)
            }
            ContentKind::TimeAndSales => {
                let config = settings
                    .visual_config
                    .clone()
                    .and_then(|cfg| cfg.time_and_sales());
                let c = Content::TimeAndSales(Some(TimeAndSales::new(
                    config,
                    derived_plan.ticker_info,
                )));

                let temp = PaneSetup {
                    push_freq: exchange::PushFrequency::ServerDefault,
                    ..derived_plan
                };

                let s = vec![trades_stream(&temp)];

                (c, s)
            }
            ContentKind::Ladder => {
                let config = settings.visual_config.clone().and_then(|cfg| cfg.ladder());
                let c = Content::Ladder(Some(Ladder::new(
                    config,
                    derived_plan.ticker_info,
                    derived_plan.price_step,
                )));

                let s = vec![depth_stream(&derived_plan), trades_stream(&derived_plan)];

                (c, s)
            }
            ContentKind::ComparisonChart => {
                let config = settings
                    .visual_config
                    .clone()
                    .and_then(|cfg| cfg.comparison());
                let basis = derived_plan.basis.unwrap_or(Basis::Time(Timeframe::M15));
                let c = Content::Comparison(Some(ComparisonChart::new(basis, &tickers, config)));

                let s = by_basis_default(
                    derived_plan.basis,
                    Timeframe::M15,
                    |tf| {
                        tickers
                            .iter()
                            .copied()
                            .map(|ti| kline_stream(ti, tf))
                            .collect()
                    },
                    || todo!("WIP: ComparisonChart does not support tick basis"),
                    |threshold| {
                        tickers
                            .iter()
                            .copied()
                            .map(|ti| StreamKind::OdbKline {
                                ticker_info: ti,
                                threshold_dbps: threshold,
                            })
                            .collect()
                    },
                );

                (c, s)
            }
            ContentKind::Starter => unreachable!(),
        }
    };

    *content = new_content;
    *streams = ResolvedStream::Ready(new_streams.clone());

    new_streams
}

/// Dispatches on basis (Time / Tick / Odb) with a default timeframe fallback.
pub(super) fn by_basis_default<T>(
    basis: Option<Basis>,
    default_tf: Timeframe,
    on_time: impl FnOnce(Timeframe) -> T,
    on_tick: impl FnOnce() -> T,
    on_odb: impl FnOnce(u32) -> T,
) -> T {
    match basis.unwrap_or(Basis::Time(default_tf)) {
        Basis::Time(tf) => on_time(tf),
        Basis::Tick(_) => on_tick(),
        Basis::Odb(threshold) => on_odb(threshold),
    }
}

/// Applies a ticksize change to content and streams.
///
/// Extracted from the `TicksizeSelected` arm in `State::update()`.
pub(super) fn apply_ticksize_change(
    content: &mut Content,
    streams: &mut ResolvedStream,
    _settings: &Settings,
    tm: TickMultiplier,
    ticker: Option<TickerInfo>,
) -> Option<Effect> {
    if let Some(ticker) = ticker {
        match content {
            Content::Kline { chart: Some(c), .. } => {
                c.change_tick_size(tm.multiply_with_min_tick_step(ticker).to_f32_lossy());
                c.reset_request_handler();
            }
            Content::Heatmap { chart: Some(c), .. } => {
                c.change_tick_size(tm.multiply_with_min_tick_step(ticker));
            }
            Content::Ladder(Some(p)) => {
                p.set_tick_size(tm.multiply_with_min_tick_step(ticker));
            }
            _ => {}
        }
    }

    let is_client = ticker
        .map(|ti| ti.exchange().is_depth_client_aggr())
        .unwrap_or(false);

    if let Some(mut it) = streams.ready_iter_mut() {
        for s in &mut it {
            if let StreamKind::Depth { depth_aggr, .. } = s {
                *depth_aggr = if is_client {
                    StreamTicksize::Client
                } else {
                    StreamTicksize::ServerSide(tm)
                };
            }
        }
    }
    if !is_client {
        Some(Effect::RefreshStreams)
    } else {
        None
    }
}

/// Applies a basis change to content and streams.
///
/// Extracted from the `BasisSelected` arm in `State::update()`.
/// The caller must set `settings.selected_basis`, reset `status`
/// and `staleness_checked` before calling this.
pub(super) fn apply_basis_change(
    content: &mut Content,
    streams: &mut ResolvedStream,
    settings: &Settings,
    new_basis: Basis,
    base_ticker: Option<TickerInfo>,
) -> Option<Effect> {
    match content {
        Content::Heatmap { chart: Some(c), .. } => {
            c.set_basis(new_basis);

            if let Some(stream_type) = streams
                .ready_iter_mut()
                .and_then(|mut it| it.find(|s| matches!(s, StreamKind::Depth { .. })))
                && let StreamKind::Depth {
                    push_freq,
                    ticker_info,
                    ..
                } = stream_type
                && ticker_info.exchange().is_custom_push_freq()
            {
                match new_basis {
                    Basis::Time(tf) => {
                        *push_freq = exchange::PushFrequency::Custom(tf);
                    }
                    Basis::Tick(_) | Basis::Odb(_) => {
                        *push_freq = exchange::PushFrequency::ServerDefault;
                    }
                }
            }

            Some(Effect::RefreshStreams)
        }
        Content::Kline { chart: Some(c), .. } => {
            if let Some(base_ticker) = base_ticker {
                match new_basis {
                    Basis::Time(tf) => {
                        let kline_stream = StreamKind::Kline {
                            ticker_info: base_ticker,
                            timeframe: tf,
                        };
                        let mut new_streams = vec![kline_stream];

                        if matches!(c.kind, data::chart::KlineChartKind::Footprint { .. }) {
                            let depth_aggr = if base_ticker.exchange().is_depth_client_aggr() {
                                StreamTicksize::Client
                            } else {
                                StreamTicksize::ServerSide(
                                    settings.tick_multiply.unwrap_or(TickMultiplier(1)),
                                )
                            };
                            new_streams.push(StreamKind::Depth {
                                ticker_info: base_ticker,
                                depth_aggr,
                                push_freq: exchange::PushFrequency::ServerDefault,
                            });
                            new_streams.push(StreamKind::Trades {
                                ticker_info: base_ticker,
                            });
                        }

                        *streams = ResolvedStream::Ready(new_streams);
                        let action = c.set_basis(new_basis);

                        if let Some(chart::Action::RequestFetch(fetch)) = action {
                            return Some(Effect::RequestFetch(fetch));
                        }
                        None
                    }
                    Basis::Tick(_) => {
                        let depth_aggr = if base_ticker.exchange().is_depth_client_aggr() {
                            StreamTicksize::Client
                        } else {
                            StreamTicksize::ServerSide(
                                settings.tick_multiply.unwrap_or(TickMultiplier(1)),
                            )
                        };

                        *streams = ResolvedStream::Ready(vec![
                            StreamKind::Depth {
                                ticker_info: base_ticker,
                                depth_aggr,
                                push_freq: exchange::PushFrequency::ServerDefault,
                            },
                            StreamKind::Trades {
                                ticker_info: base_ticker,
                            },
                        ]);
                        let _ = c.set_basis(new_basis);
                        Some(Effect::RefreshStreams)
                    }
                    Basis::Odb(threshold) => {
                        let rb_stream = StreamKind::OdbKline {
                            ticker_info: base_ticker,
                            threshold_dbps: threshold,
                        };
                        let depth_aggr = if base_ticker.exchange().is_depth_client_aggr() {
                            StreamTicksize::Client
                        } else {
                            StreamTicksize::ServerSide(
                                settings.tick_multiply.unwrap_or(TickMultiplier(1)),
                            )
                        };
                        let new_streams = vec![
                            rb_stream,
                            StreamKind::Depth {
                                ticker_info: base_ticker,
                                depth_aggr,
                                push_freq: exchange::PushFrequency::ServerDefault,
                            },
                            StreamKind::Trades {
                                ticker_info: base_ticker,
                            },
                        ];

                        *streams = ResolvedStream::Ready(new_streams);
                        let action = c.set_basis(new_basis);

                        if let Some(chart::Action::RequestFetch(fetch)) = action {
                            return Some(Effect::RequestFetch(fetch));
                        }
                        None
                    }
                }
            } else {
                None
            }
        }
        Content::Comparison(Some(c)) => match new_basis {
            Basis::Time(tf) => {
                let new_streams: Vec<StreamKind> = c
                    .selected_tickers()
                    .iter()
                    .copied()
                    .map(|ti| StreamKind::Kline {
                        ticker_info: ti,
                        timeframe: tf,
                    })
                    .collect();

                *streams = ResolvedStream::Ready(new_streams);
                let action = c.set_basis(new_basis);

                if let Some(chart::Action::RequestFetch(fetch)) = action {
                    return Some(Effect::RequestFetch(fetch));
                }
                None
            }
            Basis::Odb(threshold) => {
                let new_streams: Vec<StreamKind> = c
                    .selected_tickers()
                    .iter()
                    .copied()
                    .map(|ti| StreamKind::OdbKline {
                        ticker_info: ti,
                        threshold_dbps: threshold,
                    })
                    .collect();

                *streams = ResolvedStream::Ready(new_streams);
                let action = c.set_basis(new_basis);

                if let Some(chart::Action::RequestFetch(fetch)) = action {
                    return Some(Effect::RequestFetch(fetch));
                }
                None
            }
            _ => None,
        },
        _ => None,
    }
}

// --- Data insertion helpers ---
// These operate solely on Content but are co-located with stream wiring
// because they are called during the stream data flow pipeline.

pub(super) fn insert_hist_oi(
    content: &mut Content,
    req_id: Option<uuid::Uuid>,
    oi: &[OpenInterest],
) {
    match content {
        Content::Kline { chart, .. } => {
            let Some(chart) = chart else {
                log::warn!("insert_hist_oi: chart not yet initialized, dropping OI data");
                return;
            };
            chart.insert_open_interest(req_id, oi);
        }
        _ => {
            log::error!("pane content not candlestick");
            exchange::tg_alert!(
                exchange::telegram::Severity::Warning,
                "pane",
                "Pane content mismatch: expected candlestick"
            );
        }
    }
}

pub(super) fn insert_hist_klines(
    content: &mut Content,
    req_id: Option<uuid::Uuid>,
    timeframe: Timeframe,
    ticker_info: TickerInfo,
    klines: &[Kline],
) {
    match content {
        Content::Kline {
            chart, indicators, ..
        } => {
            let Some(chart) = chart else {
                log::warn!("insert_hist_klines: chart not yet initialized, dropping kline data");
                exchange::tg_alert!(
                    exchange::telegram::Severity::Info,
                    "pane",
                    "Chart not yet initialized for klines insert"
                );
                return;
            };

            if let Some(id) = req_id {
                if chart.basis() != Basis::Time(timeframe) {
                    log::warn!(
                        "Ignoring stale kline fetch for timeframe {:?}; \
                         chart basis = {:?}",
                        timeframe,
                        chart.basis()
                    );
                    return;
                }
                chart.insert_hist_klines(id, klines);
            } else {
                let (raw_trades, tick_size) = (chart.raw_trades(), chart.tick_size());
                let layout = chart.chart_layout();
                // GitHub Issue:
                // https://github.com/terrylica/opendeviationbar-py/issues/97
                let saved_config = chart.kline_config; // Config: Copy

                *chart = KlineChart::new(
                    layout,
                    Basis::Time(timeframe),
                    tick_size,
                    klines,
                    raw_trades,
                    indicators,
                    ticker_info,
                    chart.kind(),
                    saved_config,
                );
            }
        }
        Content::Comparison(chart) => {
            let Some(chart) = chart else {
                log::warn!(
                    "insert_hist_klines: comparison chart not yet initialized, \
                     dropping kline data"
                );
                return;
            };

            if let Some(id) = req_id {
                if chart.timeframe != timeframe {
                    log::warn!(
                        "Ignoring stale kline fetch for timeframe {:?}; \
                         chart timeframe = {:?}",
                        timeframe,
                        chart.timeframe
                    );
                    return;
                }
                chart.insert_history(id, ticker_info, klines);
            } else {
                *chart = ComparisonChart::new(
                    Basis::Time(timeframe),
                    &[ticker_info],
                    Some(chart.serializable_config()),
                );
            }
        }
        _ => {
            log::error!("pane content not candlestick or footprint");
            exchange::tg_alert!(
                exchange::telegram::Severity::Warning,
                "pane",
                "Pane content mismatch: expected candlestick/footprint"
            );
        }
    }
}

pub(super) fn insert_odb_klines(
    content: &mut Content,
    req_id: Option<uuid::Uuid>,
    ticker_info: TickerInfo,
    klines: &[Kline],
    microstructure: Option<&[Option<exchange::adapter::clickhouse::ChMicrostructure>]>,
    agg_trade_id_ranges: Option<&[Option<(u64, u64)>]>,
    open_time_ms_list: Option<&[Option<u64>]>,
) {
    match content {
        Content::Kline {
            chart, indicators, ..
        } => {
            let Some(chart) = chart else {
                log::warn!(
                    "insert_odb_klines: chart not yet initialized, \
                     dropping ODB data"
                );
                return;
            };

            if let Some(id) = req_id {
                // Historical data load -- prepend older klines to TickAggr
                chart.insert_odb_hist_klines(
                    id,
                    klines,
                    microstructure,
                    agg_trade_id_ranges,
                    open_time_ms_list,
                );
            } else {
                let (raw_trades, tick_size) = (chart.raw_trades(), chart.tick_size());
                let layout = chart.chart_layout();
                let basis = chart.basis();
                let kind = chart.kind().clone();
                // GitHub Issue:
                // https://github.com/terrylica/opendeviationbar-py/issues/97
                let saved_config = chart.kline_config; // Config: Copy

                *chart = KlineChart::new_with_microstructure(
                    layout,
                    basis,
                    tick_size,
                    klines,
                    raw_trades,
                    indicators,
                    ticker_info,
                    &kind,
                    microstructure,
                    agg_trade_id_ranges,
                    open_time_ms_list,
                    saved_config,
                );
            }
        }
        _ => {
            log::error!("pane content not candlestick for ODB klines");
            exchange::tg_alert!(
                exchange::telegram::Severity::Warning,
                "pane",
                "Pane content mismatch: expected candlestick for ODB"
            );
        }
    }
}

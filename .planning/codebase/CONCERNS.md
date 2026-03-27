# Codebase Concerns

**Analysis Date:** 2026-03-26

## Tech Debt

### SSE Timestamp Unit Mismatch (Critical Path Issue)

**Issue:** The ODB SSE sidecar stream uses `_ms` field names in `OdbBar` struct, but ClickHouse has migrated to microsecond (`_us`) precision since v13.49.

**Files:** `exchange/src/adapter/clickhouse.rs` (lines 839-872, SSE conversion)

**Impact:**

- SSE timestamp handling may have rounding errors or precision loss
- Gap between CH microsecond precision and SSE millisecond representation could cause replay edge cases
- SSE is currently **disabled by default** (FLOWSURFACE_SSE_ENABLED defaults to false) to avoid incorrect bar times

**Current Mitigation:**

- SSE feature is opt-in only via `FLOWSURFACE_SSE_ENABLED=true`
- Polling (`connect_kline_stream`) is the primary data path and uses correct CH µs→ms conversion

**Fix Approach:**

- Coordinate with upstream opendeviationbar-py to align SSE timestamp granularity with ClickHouse (upgrade OdbBar crate to use `_us` fields)
- Update `odb_bar_to_kline_tuple()` and `connect_sse_stream()` when upstream is fixed
- Add integration test for SSE vs CH timestamp reconciliation once enabled

---

### Monolithic ClickHouse Adapter

**Issue:** All ClickHouse functionality (HTTP polling, SSE streaming, SQL building, catchup gap-fill) is contained in a single 1419-line file.

**Files:** `exchange/src/adapter/clickhouse.rs`

**Impact:**

- High cognitive load for understanding the adapter
- Difficult to test individual concerns (SQL building, HTTP retry logic, SSE reconnect, gap-fill)
- Comment at line 1 acknowledges this: "FILE-SIZE-OK: monolithic adapter — CH HTTP, SSE, catchup, SQL builder are tightly coupled"

**Fix Approach:**

- Extract SQL builder into separate module (pure functions, high test coverage)
- Extract SSE stream logic into submodule with own reconnect loop
- Extract gap-fill catchup into separate module with its own timeout/retry
- HTTP retry logic already extracted to `resilience.rs`, reuse patterns there

---

### Fork-Specific State Threading Through Event Stream

**Issue:** ODB-specific data (microstructure, agg_trade_id_range, open_time_ms) must be threaded through 6 separate layers: fetch → FetchedData → KlineReceived event → pane → chart → renderer.

**Files:**

- `src/connector/fetcher.rs` (lines 28-34) — FetchedData has 3 optional fork fields
- `exchange/src/adapter.rs` — Event::KlineReceived 6th field (Option<u64>)
- `src/screen/dashboard/pane.rs` — stream handling
- `src/chart/kline/mod.rs` — bar insertion logic

**Impact:**

- Risk of field misalignment on upstream merges (6 different sites must be updated together)
- No compile-time guarantee that fork fields are propagated end-to-end
- Non-ODB adapters must always pass None, easy to miss

**Current Mitigation:**

- Merge checklist in CLAUDE.md covers these sites
- KlineReceived signature has explicit Option fields, making presence/absence visible

**Fix Approach:**

- Create wrapper enum for KlineReceived that explicitly marks fork vs upstream fields
- Add compile-time check macro that verifies all 6 threading sites are in sync
- Unit test for "fork field round-trip" across fetch→event→pane→chart

---

## Known Bugs

### Issue #17: WebSocket Reconnection Logic Under High Concurrency

**Symptoms:** Trade feed occasionally stalls for 45s+ under simultaneous Binance depth + trade reconnections. 45s is the `WS_READ_TIMEOUT`, but the watchdog should trigger a reconnect within 90s.

**Files:**

- `exchange/src/connect.rs` (line 31) — WS_READ_TIMEOUT = 45s
- `src/chart/kline/mod.rs` — independent 90s watchdog in `invalidate()`
- `exchange/src/adapter/binance.rs` — WebSocket loop with resilience::reconnect_backoff()

**Trigger:** Two+ streams attempting to reconnect simultaneously (e.g., `@aggTrade` + `@depth`) with same backoff sequence, leading to thundering herd of reconnect attempts.

**Current Behavior:**

- Backoff has jitter (`resilience::reconnect_backoff()` via `backon` crate), but jitter is applied independently per stream
- Multiple streams may still converge on same reconnect window
- No coordination between depth + trade reconnects

**Workaround:** Manual restart: `kill -9 $(pgrep -f "flowsurface.bin"); sleep 1; open Flowsurface.app`

**Fix Approach:**

- Deduplicate WebSocket connections across stream types (one WS for depth+trade together if possible per Binance API)
- Implement shared reconnect backoff counter per exchange venue (not per stream)
- Add exponential jitter offset per stream to space out reconnect attempts

---

### Issue #12: Agg Trade ID Continuity Fence

**Symptoms:** Large gaps in agg_trade_id appear when local WS trades are replayed during ODB bar completion. Sentinel (`audit_bar_continuity`) detects gaps and queues a refetch, but the refetch may retrieve old CH data if the sidecar hasn't committed the latest trades.

**Files:**

- `src/chart/kline/mod.rs` — sentinel refetch trigger
- `exchange/src/adapter/clickhouse.rs` (lines 1004-1014) — catchup response may have `partial=true`

**Impact:**

- Bar shows correct agg_trade_id range visually, but trades were missing → OFI/TradeIntensity indicators incorrect
- Sentinel logs "found N gaps" but user doesn't know the bar is stale
- No user-visible indicator that repair is pending

**Current Behavior:**

- Sentinel detects gap: logs WARN, fires Telegram alert, adds to `sentinel_refetch_pending` queue
- Refetch runs after 60s cooldown
- If sidecar still hasn't committed, refetch returns same/incomplete result
- No explicit status indicator for "waiting for sidecar commit"

**Fix Approach:**

- Implement "through_agg_id fence" — track the highest agg_trade_id we've committed to disk
- Refetch only triggers if `through_agg_id >= gap_start_id` (sidecar has newer data)
- Add visual indicator in chart legend: "[SENTINEL] Repairing agg_id gaps..." when refetch is pending
- Implement backoff: refetch delays with exponential backoff if partials keep arriving

---

### Issue #16: Forming Bar Shows from SSE Before Completing

**Symptoms:** When SSE is enabled, a forming bar appears on screen mid-trade while `Trades` stream is still accumulating ticks, causing the visual bar to jump/resize as new trades arrive.

**Files:**

- `exchange/src/adapter/clickhouse.rs` (lines 919-951) — SSE bar arrival
- `src/chart/kline/mod.rs` — forming bar rendering
- `src/screen/dashboard/pane.rs` (resolve_content streams) — must have both OdbKline + Trades streams

**Current Status:** Marked as "unblocked" in MEMORY.md — the technical constraint (needing simultaneous OdbKline+Trades streams) is already met. The feature requires the merged bar appearance logic.

**Fix Approach:**

- Implement "SSE forming bar queue" — buffer SSE forming bars and only render them once `Trades` accumulation catches up
- Suppress visual update of forming bar while `is_forming_bar && trades_are_still_arriving`

---

## Security Considerations

### ClickHouse Schema SQL Injection Vector (Low Risk)

**Risk:** `build_odb_sql()` uses string interpolation for symbol and threshold values, not parameterized queries.

**Files:** `exchange/src/adapter/clickhouse.rs` (lines 416-469)

**Current Mitigation:**

- Symbol comes from internal `TickerInfo` (exchange enum, not user-supplied)
- Threshold is u32 enum variant from upstream, not user input
- No URL encoding of query body (HTTP POST)
- ClickHouse has no prepared statement API in HTTP interface (statement caching only)

**Actual Risk:** Very low — symbol/threshold are trusted enums. But SQL construction is visible in logs, making it easy to accidentally accept user input in the future.

**Recommendations:**

- Add assertion in `build_odb_sql()` that symbol matches `[A-Z0-9]+` whitelist
- Document that symbol/threshold must come from trusted enums only
- Never accept user-supplied symbol strings

---

### Environment Variable Secrets Not Rotated

**Risk:** Telegram bot token (`FLOWSURFACE_TG_BOT_TOKEN`) and chat ID stored in `.mise.toml` plaintext.

**Files:**

- `.mise.toml` — defines environment variables
- Logged at startup if telemetry is enabled

**Current Mitigation:**

- `.mise.toml` is local development file, not committed
- Secrets are masked in Telegram alert logs

**Recommendations:**

- Use `~/.config/flowsurface/secrets.toml` (gitignored) for sensitive env vars
- Load secrets at runtime, not in `.mise.toml`
- Rotate `FLOWSURFACE_TG_BOT_TOKEN` if exposed

---

### Panic Handler Uses Blocking Telegram

**Risk:** Panic handler calls `send_alert_blocking()` which spawns a one-shot Tokio runtime. If the main runtime is corrupted, this may deadlock or escalate the panic.

**Files:** `src/main.rs` (lines 72-78)

**Current Mitigation:**

- Check `is_configured()` before attempting send
- One-shot runtime is created outside main runtime context
- Panic is still logged locally (stderr + log file)

**Recommendations:**

- Set a timeout on the one-shot runtime (e.g., 5s)
- If send times out, exit immediately (don't try to recover)
- Add telemetry flag for panic fire-and-forget (don't block shutdown)

---

## Performance Bottlenecks

### Trade Intensity Heatmap Medcouple Computation (O(n²))

**Problem:** The `trade_intensity_heatmap.rs` indicator computes medcouple (skewness measure) on every `rebuild_from_source()` using a naive O(n²) algorithm.

**Files:** `src/chart/indicator/kline/trade_intensity_heatmap.rs` (line 711 total, Medcouple impl)

**Current Impact:**

- With 7000-bar lookback window, Medcouple = O(7000²) = 49M operations
- Happens on every chart redraw or bar insertion
- Mitigated: only computed once per rebuild, not every frame

**Measurements:**

- Rebuild ~2ms locally on M3 Max (fast 7000-bar recomputation)
- Acceptable for initial load, but cumulative over session

**Fix Approach:**

- Implement efficient Medcouple (e.g., divide-and-conquer in O(n log n))
- Cache medcouple result; only recompute when window changes
- Implement incremental Medcouple update (remove oldest value, add newest) — O(log n)

---

### ClickHouse Initial Fetch Adaptive Limit Hardcoded

**Problem:** `build_odb_sql()` uses hardcoded scaling factor (20K for BPR25) to determine initial LIMIT, but this is not dynamically adjusted based on actual data density.

**Files:** `exchange/src/adapter/clickhouse.rs` (lines 426-434)

**Impact:**

- If ClickHouse schema adds more symbols or thresholds, fetch may return too few/many bars
- Intensity heatmap K=19 threshold (requires 6332+ bars) may not be reached if density increases
- No way to force full schema scan; always limited to adaptive_limit

**Fix Approach:**

- Add `LIMIT 0` metadata query to get total bar count before full fetch
- Calculate adaptive_limit based on actual data density and lookback slider setting
- Allow manual override: CLI flag or UI setting for "fetch N bars" instead of auto-scaling

---

## Fragile Areas

### ODB Bar Reconciliation: CH vs SSE vs Local Accumulation

**Files:**

- `src/chart/kline/mod.rs` (lines 1510 total) — bar insertion, replacement, reconciliation
- `exchange/src/adapter/clickhouse.rs` (lines 578-814) — polling loop updates last_ts
- Stream dispatch in pane.rs

**Why fragile:**
Three independent data sources for the same bar:

1. **ClickHouse (authoritative)** — poll every 5s, completeness guaranteed
2. **SSE (live)** — forming bar arrives mid-stream, may be incomplete
3. **Local WS trades** — accumulated in processor, completion time unknown

**Current logic:**

- SSE + local bars compete for "latest bar" slot
- CH poll replaces with authoritative version
- But if poll and SSE arrive out-of-order, stale SSE bar may overwrite fresh CH bar

**Safe modification:**

1. Always prioritize CH (most recent close_time_us wins)
2. Use agg_trade_id_range to detect which source is more complete
3. Log all reconciliation decisions at DEBUG level
4. Add test: verify final bar matches CH, not SSE

**Test coverage gaps:**

- No integration test for "CH replaces SSE" scenario
- No test for "local trades cause forming bar to complete, then SSE arrives with older timestamp"
- No test for "gap-fill trade arrives after CH bar already inserted"

---

### Pane Stream Resolution (Multiple Implicit Dependencies)

**Files:** `src/screen/dashboard/pane.rs` (lines 1-2425, resolve_content)

**Why fragile:**
ODB panes require **3 concurrent streams**:

1. `OdbKline` — ClickHouse polling
2. `Trades` — Binance WebSocket
3. `Depth` — Binance WebSocket

If any stream is missing, the pane silently waits. No error message if developer forgets to add a stream.

**Known issue:** Missing `Trades` stream causes "Waiting for trades..." forever (trade replay never triggers).

**Safe modification:**

1. Create enum `PaneStreamRequirement { optional, required }`
2. Validate pane setup at initialization: panic if required streams missing
3. Add assertion in `resolve_content()`: OdbChart must have all 3 streams
4. Log stream subscription at pane creation time

**Test coverage:** No test for "incomplete stream set" scenario.

---

### ClickHouse Connection Lifetime

**Files:**

- `exchange/src/adapter/clickhouse.rs` (line 259) — HTTP_CLIENT static LazyLock
- SSH tunnel in `.mise/tasks/infra.toml`

**Why fragile:**

- HTTP client is created once, reused forever
- SSH tunnel is external process (`ssh -L`)
- If tunnel dies, HTTP client keeps trying with 30s timeout × 3 retries = 90s per request before logging
- User has no signal that tunnel is dead until explicit Telegram alert fires

**Current mitigation:**

- Preflight script checks tunnel + ClickHouse connectivity before app launch
- If tunnel dies mid-session, `query()` retries with exponential backoff
- Telegram alerts on critical errors

**Safe modification:**

- Add periodic "tunnel health check" task (e.g., every 30s)
- Restart tunnel automatically if health check fails
- Add visual indicator in chart legend: "CH: ❌ tunnel down" when unavailable

---

## Scaling Limits

### TickAggr Vec Append on Every Trade (Vector Reallocation)

**Problem:** `TickAggr::datapoints: Vec<TickAccumulation>` grows by one on every trade arrival. For markets with 10+ trades/sec, this causes frequent reallocation.

**Files:** `data/src/aggr/ticks.rs` (lines 120+)

**Current capacity:**

- Vec starts with no pre-allocated capacity
- Reaches ~100 datapoints on busy ODB bars (1000-10000 trades per bar)
- Reallocation happens at powers of 2 (0→1→2→4→8→16→32→64→128)

**Impact:** Negligible on modern hardware, but unnecessary work for large windows.

**Fix approach:**

- Pre-allocate `Vec::with_capacity(expected_count)` based on ODB threshold estimate
- For ODB bars: pre-allocate capacity = `estimated_duration_ms / avg_trade_interval_ms`
- For tick bars: pre-allocate = tick_count

---

### ClickHouse LIMIT 2000 Scroll-Left Pagination

**Problem:** Scroll-left paginated fetches use `LIMIT 2000`. If user scrolls far left in a 7000-bar window, multiple 2000-bar fetches may be required, creating round-trip latency.

**Files:** `exchange/src/adapter/clickhouse.rs` (line 464)

**Current:** With 7000-bar limit on initial load, user can scroll ~3.5K bars left before hitting pagination. Each fetch = ~1s round-trip.

**Fix approach:**

- Use adaptive LIMIT based on scroll speed (fast scroll → larger batches)
- Or: increase LIMIT to 5000 for scroll-left (still <1MB response at high resolution)
- Cache previously-fetched pagination batches to avoid re-fetching

---

### SSE Memory Leaks on Repeated Reconnections

**Problem:** Each SSE reconnection creates a new `OdbSseClient` and `stream`. If reconnect attempts are frequent, old streams may not be dropped cleanly.

**Files:** `exchange/src/adapter/clickhouse.rs` (lines 903-911, client creation in loop)

**Current:** Loop creates new client/stream on every reconnection attempt. Previous stream should be dropped when loop continues, but no explicit cleanup.

**Fix approach:**

- Explicitly close stream before next reconnect: `drop(stream)`
- Use connection pool with max concurrent connections limit
- Add memory usage telemetry per SSE stream lifetime

---

## Dependencies at Risk

### opendeviationbar-core Semver Range: ">=13"

**Risk:** Crate is pinned at 13.55.0 but allows any patch version via `opendeviationbar-core = ">=13"` in `Cargo.toml`.

**Files:** `Cargo.toml` (opendeviationbar-core dependency)

**Impact:**

- MEMORY.md documents 32 minor versions tested (13.19→13.55), zero API breakage
- But a future major version (v14) could break bar serialization
- No changelog review process before updating

**Current mitigation:** Pinned at 13.55.0 in lock file; `cargo update` would need explicit approval.

**Fix approach:**

- Change to `opendeviationbar-core = "^13"` (allow minor.patch, restrict major)
- Review upstream changelog before any dependency bump
- Test gap-fill integration thoroughly before releasing

---

### iced GUI Framework: Upstream Can Break Canvas API

**Risk:** iced 0.14 canvas API is unstable. Upstream changes to `Canvas`, `Frame`, or `Renderer` could require major refactoring.

**Files:**

- `src/chart/kline/mod.rs` (2388 lines) — relies heavily on Canvas geometry
- `src/chart/heatmap.rs` (1042 lines)
- `src/widget/chart/comparison.rs` (1844 lines)

**Current:** Upstream is responsive to PRs, but no API stability guarantee.

**Fix approach:**

- Write Canvas abstraction layer that abstracts iced::canvas specifics
- Keep geometry computations in pure Rust (not coupled to iced)
- Plan for iced 0.15+ migration when it arrives

---

## Missing Critical Features

### No Replay Validation for Bar-Boundary Trades

**Problem:** When local WS trades are replayed into a fresh bar processor after CH reconciliation, there's no validation that replay produced the same bar as CH.

**Files:** `src/chart/kline/mod.rs` — trade_boundary_replay() call site

**Impact:** Silent discrepancies between CH and replayed bar (e.g., OHLC differs due to trade reordering).

**Fix approach:**

- After replay, compare (replayed bar OHLC) vs (CH bar OHLC)
- If mismatch > 0.01%, log ERROR and flag bar as [STALE-REPLAY]
- Fire Telegram alert with mismatch details

---

### No Per-Bar Audit Trail

**Problem:** When a bar is reconciled (CH replaces local, or sentinel refetch triggers), there's no persistent record of which source won and why.

**Files:** `src/chart/kline/mod.rs`, sentinel logic

**Impact:** Impossible to debug "why did this bar's OHLC change?" after the fact.

**Fix approach:**

- Keep audit log: `bar_id → [created_at(local), replaced_at(CH), refetched_at(sentinel)]`
- Serialize to `saved-state.json` as optional debug field
- Add CLI `--audit-bars` to print audit trail on shutdown

---

## Test Coverage Gaps

### ODB Bar Completeness: Threshold Rounding Edge Cases

**What's not tested:** Bars that complete exactly at threshold due to floating-point rounding.

**Files:** `data/src/aggr/ticks.rs` (lines 94-105, `is_full_odb()`)

**Risk:** Off-by-one in comparison: `diff * 1_000_000 >= dbps * open` may have precision issues with extreme prices.

**Missing test:**

- Unit test: very high price (e.g., BTC at $100K), very low threshold (100 dbps), verify bar closes at exact threshold
- Property test: fuzz with random prices/thresholds, verify deterministic completion

---

### Gap-Fill Trade Deduplication

**What's not tested:** Deduplication fence when gap-fill trades overlap with live WS trades.

**Files:** `src/chart/kline/mod.rs` — gap-fill insertion logic

**Risk:** Same trade inserted twice (different agg_trade_id but same price/qty/time), breaking OFI calculation.

**Missing test:**

- Integration test: fetch gap-fill, insert trades, verify no duplicates by agg_trade_id
- Test fence: trades with `agg_id <= fence` are skipped

---

### SSE Orphan Bar Filtering

**What's not tested:** Orphan bars at UTC midnight boundaries are correctly filtered.

**Files:** `exchange/src/adapter/clickhouse.rs` (line 924, orphan filter)

**Risk:** If filter fails, orphan bars appear in chart at day boundaries, disrupting visual continuity.

**Missing test:**

- Unit test: mock OdbBar with `is_orphan=Some(true)`, verify it's skipped
- Integration test: SSE stream with orphan bars, verify they never reach chart

---

## Summary: Priority Order

| Concern                              | Severity | Effort | Owner                        |
| ------------------------------------ | -------- | ------ | ---------------------------- |
| Issue #17 (WS reconnect under load)  | High     | Medium | Exchange adapter             |
| SSE timestamp unit mismatch          | High     | Medium | upstream opendeviationbar-py |
| ODB bar reconciliation test coverage | High     | Medium | Chart integration tests      |
| Issue #12 (agg_trade_id fence)       | Medium   | Medium | Chart refetch logic          |
| Monolithic ClickHouse adapter        | Medium   | High   | Refactoring                  |
| SSE forming bar visual jump          | Medium   | Medium | Chart rendering              |
| Pane stream validation               | Medium   | Low    | Chart module init            |
| Medcouple O(n²) computation          | Low      | High   | Indicator algorithm          |

---

_Concerns audit: 2026-03-26_

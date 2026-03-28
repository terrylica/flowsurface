---
plan: 03-01
phase: 03-bool-to-enum-cleanup
status: complete
started: 2026-03-28
completed: 2026-03-28
duration: ~15min
---

# Plan 03-01: Replace Bool Flags with Enums — Summary

## What Was Built

Replaced all 5 identified bool flag arguments with self-documenting enums across 11 files.

## Task Results

### Task 1: Targets 1, 2, 3, 5 (9 files)

- **Target 1**: `qty_in_quote_value(size_in_quote_ccy: bool)` → `unit: SizeUnit` (reused existing enum)
- **Target 2**: `ConditionalEma::update(active: bool)` → `action: EmaAction` (new enum)
- **Target 3**: heatmap `is_bid: bool` → `side: Side` (promoted existing ladder enum, added serde derives)
- **Target 5**: `ChaseTracker::update(is_bid: bool)` → `side: Side` (reused promoted enum)

### Task 2: Target 4 (3 files)

- **Target 4**: `insert_raw_trades(is_batches_done: bool)` → `progress: GapFillProgress` (new enum)
- Propagated through `insert_fetched_trades()` and construction sites in `dashboard.rs`

## Key Decisions

- Reused `SizeUnit` (already existed) instead of creating new enum for Target 1
- Promoted `Side` from `ladder.rs` to shared use rather than creating duplicate enum
- Added `Serialize, Deserialize` derives to `Side` for `OrderRun`/`CoalescingRun` compatibility (ephemeral runtime data, no persistence concern)

## Deviations

- **ofi.rs call sites**: Plan's Target 2 mentioned only `ofi_cumulative_ema.rs` as the sole call site, but `ofi.rs` also calls `ConditionalEma::update()` at 4 sites. Fixed during inline execution after agent interruption.
- **GapFillProgress re-export**: `odb_core` is a private module, so `GapFillProgress` needed re-export via `pub use odb_core::{BarGapKind, GapFillProgress}` in `kline/mod.rs`. Not anticipated in plan.

## Self-Check

- [x] `cargo clippy --all-targets -- -D warnings` passes clean
- [x] Zero remaining bool flag parameters in the 5 target functions
- [x] All call sites use enum variants (no bare true/false)
- [x] 2 atomic commits

## Commits

| Hash    | Description                                                                        |
| ------- | ---------------------------------------------------------------------------------- |
| c84cb9a | refactor(03-01): replace bool flags with enums (targets 1,2,3,5)                   |
| 60e5f32 | refactor(03-01): replace is_batches_done bool with GapFillProgress enum (target 4) |

## Key Files

### Created

- None (no new files)

### Modified

- `exchange/src/adapter.rs` — `qty_in_quote_value` signature
- `data/src/conditional_ema.rs` — `EmaAction` enum + `update` signature
- `data/src/chart/heatmap.rs` — `Side` usage in `OrderRun`, `CoalescingRun`, `process_side`, `update_price_level`
- `data/src/panel/ladder.rs` — `Side` serde derives + `ChaseTracker::update` signature
- `src/chart/heatmap.rs` — `depth_color` + render sites
- `src/chart/indicator/kline/ofi.rs` — `EmaAction` call sites
- `src/chart/indicator/kline/ofi_cumulative_ema.rs` — `EmaAction` call site
- `src/screen/dashboard/panel/timeandsales.rs` — `SizeUnit` call sites
- `src/screen/dashboard/panel/ladder.rs` — `Side::Bid`/`Side::Ask` call sites
- `src/chart/kline/odb_core.rs` — `GapFillProgress` enum + `insert_raw_trades` signature
- `src/chart/kline/mod.rs` — `GapFillProgress` re-export
- `src/screen/dashboard.rs` — `GapFillProgress` import + propagation

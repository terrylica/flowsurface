---
phase: 06-kline-data-ops-extraction
plan: 01
subsystem: infra
tags: [rust, module-extraction, kline, refactoring]

requires:
  - phase: 05-pane-event-dispatch
    provides: pane.rs split pattern (submodule extraction precedent)
provides:
  - kline/data_ops.rs submodule with 5 data operation methods
  - kline/mod.rs reduced by ~230 LOC (2390 -> 2160)
affects: [07-kline-odb-lifecycle, 08-indicator-ceremony]

tech-stack:
  added: []
  patterns:
    [
      kline submodule extraction via use super::* and impl KlineChart in child module,
    ]

key-files:
  created: [src/chart/kline/data_ops.rs]
  modified: [src/chart/kline/mod.rs]

key-decisions:
  - "pub(super) visibility for missing_data_task and calc_qty_scales (internal-only callers)"
  - "Exact code copy with no logic changes -- pure structural move"

patterns-established:
  - "data_ops.rs submodule pattern: use super::* + impl KlineChart block for data operations"

requirements-completed: [KLINE-01, VER-01, VER-02, VER-03]

duration: 13min
completed: 2026-03-28
---

# Phase 06 Plan 01: Kline Data Ops Extraction Summary

**Extracted 5 data operation methods (~230 LOC) from kline/mod.rs into kline/data_ops.rs submodule, following odb_core.rs pattern**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-28T02:05:12Z
- **Completed:** 2026-03-28T02:18:41Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `src/chart/kline/data_ops.rs` with 5 methods: `insert_hist_klines`, `insert_open_interest`, `toggle_indicator` (pub), `missing_data_task`, `calc_qty_scales` (pub(super))
- Reduced kline/mod.rs from 2390 to 2160 LOC (-230 lines)
- All 17 existing kline tests pass unchanged
- clippy clean with `-D warnings`, no unsafe code introduced

## Task Commits

Each task was committed atomically:

1. **Task 1: Create data_ops.rs with 5 extracted methods** - `9a6b167` (feat)
2. **Task 2: Update mod.rs -- declare submodule, remove extracted methods** - `2771bd9` (refactor)

## Files Created/Modified

- `src/chart/kline/data_ops.rs` - New submodule with 5 data operation methods (257 LOC)
- `src/chart/kline/mod.rs` - Added `mod data_ops;` declaration, removed 5 method bodies, updated FILE-SIZE-OK comment

## Decisions Made

- Used `pub(super)` for `missing_data_task` and `calc_qty_scales` since they are only called from within kline/mod.rs (by `invalidate()` and `draw()` respectively)
- Kept `pub` for `insert_hist_klines`, `insert_open_interest`, and `toggle_indicator` since they are called from pane code outside the kline module
- Exact code copy with no logic changes -- pure structural move

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

- Pre-existing `cargo fmt` issues exist across multiple files in the repo (dashboard.rs, clickhouse.rs, heatmap.rs, etc.) -- these are not caused by this plan's changes and are out of scope. Clippy passes clean. Only ran fmt on the two files modified by this plan.

## Known Stubs

None.

## Next Phase Readiness

- kline/mod.rs at 2160 LOC, needs ~360 more LOC reduction to reach Phase 7 target of <1800
- Phase 7 (KLINE-02: ODB lifecycle extraction) targets ~600 LOC, well above the gap
- RefCell borrow discipline in remaining mod.rs code needs careful review for Phase 7

---

_Phase: 06-kline-data-ops-extraction_
_Completed: 2026-03-28_

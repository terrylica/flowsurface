---
phase: 08-indicator-ceremony-reduction
plan: 01
subsystem: infra
tags: [rust, enum-map, indicator, factory-pattern, refactor]

# Dependency graph
requires:
  - phase: 07-kline-odb-lifecycle
    provides: kline/mod.rs split into submodules (data_ops.rs, odb_core.rs)
provides:
  - Single make_indicator(which, cfg) factory in src/chart/indicator/kline.rs
  - Verified 3-file indicator ceremony documented in CLAUDE.md
affects: [future-indicator-additions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single config-aware factory for all indicator construction"

key-files:
  created: []
  modified:
    - src/chart/indicator/kline.rs
    - src/chart/kline/mod.rs
    - src/chart/kline/data_ops.rs
    - src/chart/kline/odb_core.rs
    - src/chart/indicator/kline/ofi.rs
    - src/chart/indicator/kline/ofi_cumulative_ema.rs
    - src/chart/indicator/kline/trade_intensity_heatmap.rs
    - CLAUDE.md

key-decisions:
  - "Consolidated make_empty + make_indicator_with_config into single make_indicator(which, cfg)"
  - "Removed dead new() constructors from OFI, OFICumulativeEma, TradeIntensityHeatmap (always use config-aware constructors)"

patterns-established:
  - "Single factory: all indicator construction goes through indicator::kline::make_indicator(which, cfg)"

requirements-completed: [QUAL-03, VER-01, VER-02, VER-03]

# Metrics
duration: 2min
completed: 2026-03-28
---

# Phase 8 Plan 1: Indicator Ceremony Reduction Summary

**Consolidated dual indicator factories into single make_indicator(which, cfg) and documented verified 3-file ceremony in CLAUDE.md**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-28T03:06:00Z
- **Completed:** 2026-03-28T03:08:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Merged confusing dual-factory pattern (make_empty + make_indicator_with_config) into single make_indicator(which, cfg)
- Updated all 5 call sites across 4 files to use consolidated factory
- Removed 3 dead new() constructors that became unused after consolidation
- Replaced stale 6-step CLAUDE.md checklist with verified 3-file ceremony documentation

## Task Commits

Each task was committed atomically:

1. **Task 1: Consolidate factory functions into single make_indicator** - `bf1977c` (refactor)
2. **Task 2: Update CLAUDE.md indicator checklist** - `fffb5da` (docs)

## Files Created/Modified

- `src/chart/indicator/kline.rs` - Consolidated factory: make_empty -> make_indicator(which, cfg)
- `src/chart/kline/mod.rs` - Removed make_indicator_with_config, updated 3 call sites
- `src/chart/kline/data_ops.rs` - Updated call site to use consolidated factory
- `src/chart/kline/odb_core.rs` - Updated call site to use consolidated factory
- `src/chart/indicator/kline/ofi.rs` - Removed dead new() constructor
- `src/chart/indicator/kline/ofi_cumulative_ema.rs` - Removed dead new() constructor
- `src/chart/indicator/kline/trade_intensity_heatmap.rs` - Removed dead new() constructor
- `CLAUDE.md` - Updated "Adding a New Indicator" section with verified 3-file checklist

## Decisions Made

- Consolidated make_empty + make_indicator_with_config into single make_indicator(which, cfg) -- eliminates confusing two-function pattern where config could be silently ignored
- Removed dead new() constructors rather than keeping as convenience -- make_indicator always takes Config (which has serde defaults)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed dead new() constructors flagged by clippy**

- **Found during:** Task 1 (factory consolidation)
- **Issue:** After consolidation, OFIIndicator::new(), OFICumulativeEmaIndicator::new(), and TradeIntensityHeatmapIndicator::new() had zero callers, causing clippy -D warnings failure
- **Fix:** Removed the 3 dead new() functions; all construction now goes through with_ema_period/with_config
- **Files modified:** src/chart/indicator/kline/ofi.rs, ofi_cumulative_ema.rs, trade_intensity_heatmap.rs
- **Verification:** cargo clippy --all-targets -- -D warnings passes clean
- **Committed in:** bf1977c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug -- dead code)
**Impact on plan:** Necessary for clippy compliance. No scope creep.

## Issues Encountered

None

## Known Stubs

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 8 is the final phase in the refactoring roadmap
- All 8 phases complete: config centralization, must_use annotations, pane split, kline split, ODB lifecycle, indicator ceremony
- Codebase ready for feature development with reduced ceremony

---

_Phase: 08-indicator-ceremony-reduction_
_Completed: 2026-03-28_

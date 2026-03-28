---
phase: 04-pane-content-extraction
plan: 01
subsystem: ui
tags: [rust, module-split, god-module, pane, iced]

# Dependency graph
requires:
  - phase: 02-must-use-annotations
    provides: "#[must_use] on Action/Effect return types — safety net for code moves"
provides:
  - "Content enum in standalone pane/content.rs submodule (~465 LOC)"
  - "pane/mod.rs reduced from 2431 to 1980 LOC"
  - "pub(crate) use content::Content re-export preserving all 40+ external references"
affects: [05-stream-setup-extraction, 07-kline-odb-lifecycle]

# Tech tracking
tech-stack:
  added: []
  patterns:
    [file-to-directory module conversion, pub(super) for cross-submodule access]

key-files:
  created: [src/screen/dashboard/pane/content.rs]
  modified: [src/screen/dashboard/pane/mod.rs]

key-decisions:
  - "git mv pane.rs to pane/mod.rs preserves git history for the larger file"
  - "pub(super) for factory methods (new_heatmap, new_kline, placeholder, initialized) — minimal visibility"
  - "ComparisonChart import stays in mod.rs (used by State::insert_hist_klines and set_content_and_streams)"

patterns-established:
  - "file-to-directory conversion: git mv foo.rs foo/mod.rs + mod submodule; pub(crate) use re-export"
  - "pub(super) for methods that were private but need cross-submodule access within same parent"

requirements-completed: [PANE-01, VER-01, VER-02, VER-03]

# Metrics
duration: 5min
completed: 2026-03-28
---

# Phase 4 Plan 1: Pane Content Extraction Summary

**Content enum with 12 methods, Display and PartialEq impls extracted from 2431-LOC god module pane.rs into standalone pane/content.rs (465 LOC), reducing mod.rs to 1980 LOC**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-28T01:17:35Z
- **Completed:** 2026-03-28T01:22:42Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extracted Content enum and all 12 impl methods to pane/content.rs (465 LOC)
- Reduced pane/mod.rs from 2431 to 1980 LOC (19% reduction)
- All 40+ external pane::Content references compile unchanged via pub(crate) re-export
- Zero behavior changes — pure structural move

## Task Commits

Each task was committed atomically:

1. **Task 1: Convert pane.rs to pane/mod.rs and extract Content** - `9090e3c` (refactor)
2. **Task 2: Update FILE-SIZE-OK comment and verify final state** - `8b16880` (chore)

## Files Created/Modified

- `src/screen/dashboard/pane/content.rs` - Content enum, factory methods (new_heatmap, new_kline, placeholder), helper methods (last_tick, chart_kind, toggle_indicator, reorder_indicators, change_visual_config, studies, update_studies, kind, initialized), Display and PartialEq impls
- `src/screen/dashboard/pane/mod.rs` - Remainder: State struct, Effect/Status/Action/Message/Event enums, all State impl blocks, free functions (link_group_modal, ticksize_modifier, basis_modifier, by_basis_default)

## Decisions Made

- Used git mv for pane.rs to pane/mod.rs to preserve git history for the larger file
- Applied pub(super) visibility (not pub) for factory methods — minimal exposure principle
- ComparisonChart import kept in mod.rs since State::insert_hist_klines and set_content_and_streams reference it directly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added Indicator trait import to content.rs**

- **Found during:** Task 1 (compile check)
- **Issue:** `KlineIndicator::for_market()` requires the `Indicator` trait in scope
- **Fix:** Added `use data::chart::indicator::Indicator` to content.rs imports
- **Files modified:** src/screen/dashboard/pane/content.rs
- **Verification:** cargo clippy passes clean
- **Committed in:** 9090e3c (Task 1 commit)

**2. [Rule 3 - Blocking] Fixed import adjustments in mod.rs**

- **Found during:** Task 1 (compile check)
- **Issue:** After extraction, mod.rs had unused imports (HeatmapChart, HeatmapIndicator, KlineIndicator, ViewConfig, Indicator) and missing ComparisonChart import
- **Fix:** Removed unused imports, restored ComparisonChart import
- **Files modified:** src/screen/dashboard/pane/mod.rs
- **Verification:** cargo clippy passes clean
- **Committed in:** 9090e3c (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking — import resolution)
**Impact on plan:** Standard import adjustments expected in any module extraction. No scope creep.

## Issues Encountered

Pre-existing `cargo fmt` failures across many files (heatmap.rs, clickhouse.rs, adapter.rs, dashboard.rs, timeandsales.rs, main.rs, config.rs). These are NOT caused by this plan's changes. Only the pane/mod.rs formatting diff was fixed (collapsed multi-line import to single line). Pre-existing issues logged but not fixed per deviation scope rules.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - pure structural refactoring, no new data flows or UI elements.

## Next Phase Readiness

- pane/mod.rs is now modular with content.rs extracted
- Ready for Phase 5 stream setup extraction (further splitting mod.rs)
- FILE-SIZE-OK comment updated to reference Phase 5 plan

---

_Phase: 04-pane-content-extraction_
_Completed: 2026-03-28_

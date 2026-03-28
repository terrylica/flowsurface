---
phase: 05-pane-stream-setup-extraction
plan: 01
subsystem: ui
tags: [rust, module-extraction, pane, stream-wiring, borrow-checker]

# Dependency graph
requires:
  - phase: 04-pane-content-extraction
    provides: pane/mod.rs + pane/content.rs split with pub(super) pattern
provides:
  - pane/stream_setup.rs with 7 pub(super) free functions for stream wiring
  - pane/mod.rs reduced from 1980 to 1409 LOC
affects: [06-kline-split, 07-kline-odb-lifecycle]

# Tech tracking
tech-stack:
  added: []
  patterns: [free-functions-with-field-refs for borrow-checker-safe extraction]

key-files:
  created: [src/screen/dashboard/pane/stream_setup.rs]
  modified: [src/screen/dashboard/pane/mod.rs]

key-decisions:
  - "Free functions with individual field refs instead of &mut self methods to avoid split-borrow conflicts"
  - "Data insertion helpers (insert_hist_oi/klines/odb_klines) co-located in stream_setup.rs rather than content.rs for LOC margin"
  - "_settings prefix for unused parameter in apply_ticksize_change (kept for API symmetry with apply_basis_change)"

patterns-established:
  - "Free function extraction: pass individual &mut field refs when callers hold partial borrows on parent struct"
  - "Thin wrapper delegation: State methods delegate to stream_setup:: preserving public API unchanged"

requirements-completed: [PANE-02, PANE-03, VER-01, VER-02, VER-03]

# Metrics
duration: 5min
completed: 2026-03-28
---

# Phase 05 Plan 01: Stream Setup Extraction Summary

**Extracted 7 stream wiring functions from pane/mod.rs (1980 LOC) to pane/stream_setup.rs (702 LOC), reducing mod.rs to 1409 LOC with ODB triple-stream invariant preserved across all 4 construction sites**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-28T01:45:34Z
- **Completed:** 2026-03-28T01:51:18Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Extracted build_content_and_streams, by_basis_default, apply_ticksize_change, apply_basis_change, insert_hist_oi, insert_hist_klines, insert_odb_klines to stream_setup.rs
- Reduced pane/mod.rs from 1980 to 1409 LOC (29% reduction, well under 1500 target)
- ODB triple-stream invariant (OdbKline + Trades + Depth) verified in all 4 construction sites plus the BasisSelected Odb arm
- Zero behavior changes -- all dashboard.rs callers work unchanged via thin wrapper delegation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create stream_setup.rs with extracted free functions** - `b0eba16` (refactor)

## Files Created/Modified

- `src/screen/dashboard/pane/stream_setup.rs` - New module with 7 pub(super) free functions for stream wiring and data insertion
- `src/screen/dashboard/pane/mod.rs` - Reduced to thin wrappers delegating to stream_setup; removed moved code and unused imports

## Decisions Made

- Free functions with individual field references (not &mut self methods) to avoid borrow checker conflicts when callers hold partial borrows on State fields (e.g., self.modal.take() in update())
- Data insertion helpers placed in stream_setup.rs rather than content.rs to ensure LOC margin (without them: ~1562 LOC, over target)
- Kept \_settings parameter in apply_ticksize_change for API symmetry even though currently unused

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored VisualConfig import removed during cleanup**

- **Found during:** Task 1 (import cleanup)
- **Issue:** VisualConfig was incorrectly removed from mod.rs imports -- still needed by Message::VisualConfigChanged enum variant
- **Fix:** Added VisualConfig back to the data::layout::pane import group
- **Files modified:** src/screen/dashboard/pane/mod.rs
- **Verification:** cargo clippy passes clean
- **Committed in:** b0eba16 (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minor import correction during cleanup. No scope creep.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Known Stubs

None

## Next Phase Readiness

- pane/mod.rs is at 1409 LOC with clear separation: State struct + view + update skeleton in mod.rs, Content in content.rs, stream wiring in stream_setup.rs
- Ready for Phase 06 (kline split) which targets kline/mod.rs god module

---

_Phase: 05-pane-stream-setup-extraction_
_Completed: 2026-03-28_

---
phase: 07-kline-odb-lifecycle-extraction
plan: 01
subsystem: chart
tags: [rust, refactoring, module-extraction, kline, odb-lifecycle]

requires:
  - phase: 06-kline-data-ops-extraction
    provides: "Established kline submodule extraction pattern (data_ops.rs)"
provides:
  - "ODB lifecycle methods in kline/odb_lifecycle.rs (watchdog, sentinel, viewport digest, telemetry)"
  - "kline/mod.rs reduced to 1721 LOC (under 1800 target)"
affects: [kline-rendering, pane-splitting]

tech-stack:
  added: []
  patterns:
    ["ODB lifecycle delegation via pub(super) methods in dedicated submodule"]

key-files:
  created: ["src/chart/kline/odb_lifecycle.rs"]
  modified: ["src/chart/kline/mod.rs"]

key-decisions:
  - "All guard conditions moved INTO helper methods for maximum LOC reduction"
  - "Relocated lifecycle-related tests (cooldown arithmetic + guard_allows) to odb_lifecycle.rs"
  - "SystemTime/UNIX_EPOCH imports kept in mod.rs for use super::* availability"

patterns-established:
  - "ODB lifecycle extraction: fire-and-forget methods called from invalidate() orchestrator"

requirements-completed: [KLINE-02, KLINE-03, VER-01, VER-02, VER-03]

duration: 7min
completed: 2026-03-28
---

# Phase 7 Plan 1: ODB Lifecycle Extraction Summary

**Extracted 4 ODB lifecycle orchestration methods (watchdog, sentinel, viewport digest, telemetry) from kline/mod.rs into odb_lifecycle.rs, reducing mod.rs from 2161 to 1721 LOC**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-28T02:38:54Z
- **Completed:** 2026-03-28T02:46:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created odb_lifecycle.rs with 4 pub(super) lifecycle methods following established extraction pattern
- Reduced kline/mod.rs from 2161 LOC to 1721 LOC (439 LOC reduction, well under 1800 target)
- Relocated 8 lifecycle-related tests (3 cooldown arithmetic + 5 guard_allows) to odb_lifecycle.rs
- cargo clippy --all-targets -- -D warnings passes clean, zero unsafe code

## Task Commits

Each task was committed atomically:

1. **Task 1: Create odb_lifecycle.rs with 4 extracted lifecycle methods + relocated tests** - `f1f6971` (refactor)
2. **Task 2: Update mod.rs -- declare submodule, replace inline blocks with method calls, remove relocated tests** - `212e1ac` (refactor)

## Files Created/Modified

- `src/chart/kline/odb_lifecycle.rs` - New submodule: ODB lifecycle orchestration (watchdog, sentinel audit, viewport digest, telemetry snapshot)
- `src/chart/kline/mod.rs` - Reduced from 2161 to 1721 LOC; inline lifecycle blocks replaced with 4 method calls in invalidate()

## Decisions Made

- All guard conditions (is_odb checks, timer checks, fetching_trades checks) moved INTO the helper methods rather than kept at the call site -- maximizes LOC reduction in mod.rs
- Relocated cooldown arithmetic and guard_allows tests to odb_lifecycle.rs since they test lifecycle guard logic
- Kept SystemTime/UNIX_EPOCH imports in mod.rs so they're available to odb_lifecycle.rs via use super::\*

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- kline/mod.rs at 1721 LOC -- ready for Phase 8 (indicator ceremony reduction) or further extraction
- All 5 kline submodules (bar_selection, crosshair, data_ops, odb_core, odb_lifecycle) follow consistent use super::\* pattern
- invalidate() is now a clean orchestrator: autoscale/cache-clear + 4 lifecycle method calls + missing_data_task return

---

_Phase: 07-kline-odb-lifecycle-extraction_
_Completed: 2026-03-28_

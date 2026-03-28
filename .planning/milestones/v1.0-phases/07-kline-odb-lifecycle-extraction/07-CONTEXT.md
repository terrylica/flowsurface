# Phase 7: Kline ODB Lifecycle Extraction - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

ODB orchestration (gap-fill, reconciliation, sentinel, watchdog) extracted from kline/mod.rs to kline/odb_lifecycle.rs. Fork-specific ODB complexity isolated in one file. kline/mod.rs drops below 1800 LOC. No new RefCell borrow sites introduced.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase. CRITICAL: Preserve existing RefCell borrow discipline. Never hold an immutable borrow across a borrow_mut() call. Runtime panics, not compile errors.

</decisions>

<code_context>

## Existing Code Insights

- kline/mod.rs is currently ~2160 LOC (after Phase 6 data_ops extraction)
- odb_core.rs already exists as a submodule handling ODB-specific core logic
- data_ops.rs already exists as a submodule handling data insertion
- Phase 7 needs to move ~360+ LOC to get under 1800

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>

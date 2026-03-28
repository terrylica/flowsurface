# Phase 4: Pane Content Extraction - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Content enum and its factory methods live in their own file -- pane.rs loses ~500 LOC and Content becomes independently navigable. Extract to src/screen/dashboard/pane/content.rs.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions. Key constraint: pane/mod.rs must re-export Content via `pub(crate) use content::Content` so all external imports remain unchanged.

</decisions>

<code_context>

## Existing Code Insights

Codebase context will be gathered during plan-phase research.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>

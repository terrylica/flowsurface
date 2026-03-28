# Phase 8: Indicator Ceremony Reduction - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Adding a new indicator requires touching 6 or fewer files instead of 36. The highest-leverage change for future development velocity. Must use explicit match arms (compiler-checked), not trait objects or macros.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase. Key constraints from ROADMAP success criteria:

1. A documented checklist exists showing exactly which files to touch for a new indicator (6 or fewer)
2. The indicator registration path uses explicit match arms (not trait objects or macros) -- compiler-checked, not magic
3. FOR_SPOT and FOR_PERPS arrays remain as explicit lists

</decisions>

<code_context>

## Existing Code Insights

Codebase context will be gathered during plan-phase research. The RSI indicator addition (commit 0d21c59) touched 36 files — research should analyze which of those touches were structural coupling (now resolved by Phases 1-7) vs essential.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>

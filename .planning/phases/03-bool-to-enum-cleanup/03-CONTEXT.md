# Phase 3: Bool-to-Enum Cleanup - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Bool flag arguments replaced with self-documenting enums -- call sites read like prose instead of `true/false` mystery flags. Targets 5 identified bool flag functions across adapter.rs, conditional_ema.rs, heatmap.rs, odb_core.rs, and ladder.rs.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions. Enum names should be descriptive and follow Rust naming conventions (PascalCase variants).

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

# Phase 5: Pane Stream Setup Extraction - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Stream wiring logic (resolve_content, set_content_and_streams) extracted from pane/mod.rs to pane/stream_setup.rs. pane/mod.rs drops below 1500 LOC. Stream setup becomes independently reviewable. CRITICAL: ODB panes must still register all 3 streams (OdbKline + Trades + Depth).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

</decisions>

<code_context>

## Existing Code Insights

Codebase context will be gathered during plan-phase research. Note: pane.rs was already split into pane/mod.rs + pane/content.rs in Phase 4.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>

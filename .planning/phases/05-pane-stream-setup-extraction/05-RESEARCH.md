# Phase 5: Pane Stream Setup Extraction - Research

**Researched:** 2026-03-27
**Domain:** Rust module extraction -- stream wiring logic from pane/mod.rs
**Confidence:** HIGH

## Summary

Phase 5 extracts stream wiring logic from `src/screen/dashboard/pane/mod.rs` (currently 1980 LOC) into a new `pane/stream_setup.rs` module. The target is getting pane/mod.rs below 1500 LOC, requiring 481+ LOC to move.

The primary extraction candidates are: `set_content_and_streams()` (228 LOC), `by_basis_default()` (13 LOC), and the `BasisSelected`/`TicksizeSelected` stream-switching logic currently embedded in the `StreamModifierChanged` event handler (~241 LOC of extractable inner logic). Together these total ~482 LOC -- just barely meeting the threshold. However, the `StreamModifierChanged` handler requires careful extraction since it mutates `self.modal`, `self.settings`, `self.streams`, `self.status`, `self.staleness_checked`, and `self.content` -- all fields on `State`. The recommended approach is to extract the inner stream-switching logic as free functions or methods that take individual field references rather than `&mut self`.

**Primary recommendation:** Extract `set_content_and_streams` as a method on State that delegates to free functions in `stream_setup.rs`, extract `by_basis_default` as a free function, and extract the `BasisSelected` and `TicksizeSelected` handler bodies as free functions that receive the specific mutable references they need (`&mut Content`, `&mut ResolvedStream`, `&Settings`, etc.).

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- all implementation choices at Claude's discretion (infrastructure phase).

### Claude's Discretion

All implementation choices are at Claude's discretion -- pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

### Deferred Ideas (OUT OF SCOPE)

None -- infrastructure phase.
</user_constraints>

<phase_requirements>

## Phase Requirements

| ID      | Description                                                                                                    | Research Support                                                                                                                                                                                                                                                                                                                                                                        |
| ------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PANE-02 | Stream setup logic (OdbKline + Trades + Depth wiring) extracted to `pane/stream_setup.rs` (~700 LOC reduction) | `set_content_and_streams` (228 LOC) + `by_basis_default` (13 LOC) + BasisSelected/TicksizeSelected handler extraction (~241 LOC) = ~482 LOC. The REQUIREMENTS.md target of ~700 was aspirational; actual extractable stream-wiring code is ~482-500 LOC, which still meets the PANE-03 threshold.                                                                                       |
| PANE-03 | pane.rs reduced below 1500 LOC after extractions                                                               | 1980 - 482 = ~1498 LOC. Tight margin. Consider also extracting `insert_hist_oi` (19 LOC), `insert_hist_klines` (88 LOC), and `insert_odb_klines` (61 LOC) as data insertion helpers -- these are logically "data operations" not "stream setup" but moving them to stream_setup.rs (or a separate data_ops.rs) provides safety margin. With these: 1980 - 650 = ~1330, well under 1500. |
| VER-01  | `cargo clippy -- -D warnings` passes after every phase                                                         | Standard verification gate                                                                                                                                                                                                                                                                                                                                                              |
| VER-02  | Zero behavior changes -- all existing functionality works identically                                          | Must verify ODB triple-stream wiring preserved                                                                                                                                                                                                                                                                                                                                          |
| VER-03  | No new `unsafe` code introduced                                                                                | No unsafe needed for this extraction                                                                                                                                                                                                                                                                                                                                                    |

</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **No regressions**: `cargo clippy -- -D warnings` must pass clean
- **No behavior change**: All existing functionality must work identically
- **Incremental**: Each phase independently shippable
- **Rust edition**: 2024, toolchain 1.93.1
- **Max line width**: 100 characters (rustfmt.toml)
- **Lint**: `mise run lint` = `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings`
- **FILE-SIZE-OK comments**: Update or remove from pane/mod.rs after extraction
- **Fork patterns**: `// NOTE(fork):` prefix for fork-specific deviations; `// GitHub Issue:` links preserved
- **Visibility**: `pub(super)` for cross-submodule access within pane/ directory (established in Phase 4)

## Architecture Patterns

### Current pane/mod.rs Structure (1980 LOC)

```
Lines 1-46:     Imports + mod content
Lines 47-108:   Enums (Effect, Status, Action, Message, Event)
Lines 109-122:  State struct definition
Lines 124-169:  State::new, from_config, stream_pair, stream_pair_kind
Lines 171-398:  State::set_content_and_streams  <<<< EXTRACT
Lines 400-569:  State::insert_hist_oi/klines/odb_klines  <<<< CANDIDATE
Lines 571-576:  State::has_stream
Lines 578-1126: State::view (549 LOC -- view/rendering, NOT extractable here)
Lines 1128-1139: State::apply_keyboard_nav
Lines 1142-1543: State::update (402 LOC -- event handler)
  - 1223-1483: StreamModifierChanged (261 LOC)  <<<< EXTRACT INNER LOGIC
Lines 1545-1643: State::view_controls
Lines 1646-1750: State::compose_stack_view
Lines 1752-1778: State::matches_stream, show_modal_with_focus
Lines 1781-1862: State::invalidate/update_interval/last_tick/tick/unique_id
Lines 1864-1878: Default impl
Lines 1880-1980: Free functions (link_group_modal, ticksize_modifier, basis_modifier, by_basis_default)
```

### Recommended Extraction Plan

**Target file:** `src/screen/dashboard/pane/stream_setup.rs`

**Tier 1 -- Clean extractions (stream setup, 241 LOC):**

| Function                  | Lines     | LOC | Extraction Approach                                                                                      |
| ------------------------- | --------- | --- | -------------------------------------------------------------------------------------------------------- |
| `set_content_and_streams` | 171-398   | 228 | Move body to free fn `build_content_and_streams()` in stream_setup.rs; State method becomes thin wrapper |
| `by_basis_default`        | 1968-1980 | 13  | Move directly as `pub(super) fn`                                                                         |

**Tier 2 -- Handler extraction (stream switching, ~241 LOC):**

The `StreamModifierChanged` handler in `update()` contains two large arms that are pure stream-wiring logic:

| Sub-handler        | Approx LOC | Approach                                                                                                                                               |
| ------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TicksizeSelected` | ~43        | Extract as `apply_ticksize_change(&mut Content, &mut ResolvedStream, &Settings, TickMultiplier, Option<TickerInfo>) -> Option<Effect>`                 |
| `BasisSelected`    | ~198       | Extract as `apply_basis_change(&mut Content, &mut ResolvedStream, &mut Settings, &mut Status, &mut bool, Basis, Option<TickerInfo>) -> Option<Effect>` |

The `update()` method retains the match skeleton but delegates to stream_setup functions. This avoids borrowing issues since we pass individual fields rather than `&mut self`.

**Tier 3 -- Data insertion helpers (buffer, ~168 LOC):**

| Function             | Lines   | LOC | Notes                           |
| -------------------- | ------- | --- | ------------------------------- |
| `insert_hist_oi`     | 400-418 | 19  | Operates only on `self.content` |
| `insert_hist_klines` | 420-507 | 88  | Operates only on `self.content` |
| `insert_odb_klines`  | 509-569 | 61  | Operates only on `self.content` |

These access only `self.content` and could be extracted as methods on `Content` in content.rs, or as free functions in stream_setup.rs. Moving them to `Content` methods in content.rs is more semantically correct since they operate solely on content. However, this violates the phase boundary (Phase 4 already defined content.rs). Pragmatic choice: move to stream_setup.rs with a comment noting they are data operations co-located with stream wiring for LOC reduction.

**LOC accounting:**

| Extraction                                 | LOC moved | Remaining |
| ------------------------------------------ | --------- | --------- |
| set_content_and_streams body               | ~220      | ~1760     |
| by_basis_default                           | ~13       | ~1747     |
| BasisSelected/TicksizeSelected inner logic | ~200      | ~1547     |
| Thin wrapper overhead added back           | +15       | ~1562     |
| insert_hist_oi/klines/odb_klines           | ~160      | ~1402     |
| Thin wrapper overhead added back           | +10       | ~1412     |

**Result: ~1412 LOC** -- safely under 1500 with margin.

Without Tier 3: ~1562 LOC -- over the 1500 target. **Tier 3 is required.**

### Recommended Module Structure

```
src/screen/dashboard/pane/
  mod.rs           # ~1412 LOC (State struct, view, update skeleton, controls)
  content.rs       # ~465 LOC (Content enum, factory methods, indicator ops)
  stream_setup.rs  # ~560 LOC (stream wiring, basis switching, data insertion)
```

### Field Access Pattern for Extracted Functions

The key challenge is that `BasisSelected` handler accesses 6 different `self` fields. In Rust, you cannot call `self.extracted_method()` while also borrowing `self.content` mutably. The solution: **pass individual field references**.

```rust
// In stream_setup.rs:
pub(super) fn apply_basis_change(
    content: &mut Content,
    streams: &mut ResolvedStream,
    settings: &Settings,
    status: &mut Status,
    staleness_checked: &mut bool,
    new_basis: Basis,
    base_ticker: Option<TickerInfo>,
) -> Option<Effect> {
    // ... extracted logic
}

// In mod.rs update():
modal::stream::Action::BasisSelected(new_basis) => {
    modifier.update_kind_with_basis(new_basis);
    self.settings.selected_basis = Some(new_basis);
    effect = stream_setup::apply_basis_change(
        &mut self.content,
        &mut self.streams,
        &self.settings,
        &mut self.status,
        &mut self.staleness_checked,
        new_basis,
        self.stream_pair(),  // called before mutable borrows
    );
}
```

This pattern avoids split-borrow issues entirely.

### Import Pattern

```rust
// In mod.rs:
mod stream_setup;

// stream_setup.rs needs:
use super::{Content, Effect, Status};
use crate::chart;
use crate::connector::ResolvedStream;
use data::{chart::Basis, layout::pane::{ContentKind, PaneSetup, Settings}};
use exchange::{
    Kline, OpenInterest, TickMultiplier, TickerInfo, Timeframe,
    adapter::{StreamKind, StreamTicksize},
};
```

## Don't Hand-Roll

| Problem           | Don't Build              | Use Instead                                         | Why                                                         |
| ----------------- | ------------------------ | --------------------------------------------------- | ----------------------------------------------------------- |
| Borrow splitting  | Complex RefCell wrappers | Pass individual `&mut` field refs to free functions | Rust's borrow checker handles field-level borrows naturally |
| Method delegation | Trait-based dispatch     | Thin wrapper methods that call module functions     | Keeps State API unchanged for dashboard.rs callers          |

## Common Pitfalls

### Pitfall 1: Split Borrow Conflicts

**What goes wrong:** Extracting `BasisSelected` as `self.apply_basis_change()` fails because the method already borrows `&mut self` while the caller needs `self.modal` mutably.
**Why it happens:** The `StreamModifierChanged` handler does `self.modal.take()` then mutates other fields. A `&mut self` method can't coexist with the outer borrow on `self.modal`.
**How to avoid:** Use free functions with individual field references, not `&mut self` methods.
**Warning signs:** Compile error "cannot borrow `*self` as mutable more than once".

### Pitfall 2: ODB Triple-Stream Invariant Broken

**What goes wrong:** After extraction, ODB panes only get 2 of 3 required streams (OdbKline + Trades + Depth).
**Why it happens:** Stream construction logic is duplicated in both `set_content_and_streams` (initial setup) and `BasisSelected` (runtime switching). Missing one site during extraction.
**How to avoid:** Grep for all `StreamKind::OdbKline` construction sites and verify each co-occurs with Trades + Depth streams. After extraction, test by opening an ODB pane.
**Warning signs:** "Waiting for trades..." in ODB pane after switching tickers.

### Pitfall 3: Stale `stream_pair()` Value

**What goes wrong:** `stream_pair()` returns the old ticker after streams have been reassigned.
**Why it happens:** `stream_pair()` reads from `self.streams` which may have just been replaced. If called after stream mutation in the extracted function, returns new value; if called before, returns old value.
**How to avoid:** In `BasisSelected`, `stream_pair()` is called before stream mutation (line 1286). The extracted function must receive the ticker as a parameter, not call stream_pair() internally.
**Warning signs:** Wrong ticker in stream subscriptions after basis switch.

### Pitfall 4: LOC Miscounting

**What goes wrong:** After extraction, pane/mod.rs is at 1510 LOC -- above threshold.
**Why it happens:** Thin wrappers and new `use` imports add lines back. The margin is tight without Tier 3 extractions.
**How to avoid:** Include Tier 3 (insert_hist_oi/klines/odb_klines) in the extraction. Verify with `wc -l` after each extraction step.
**Warning signs:** `wc -l src/screen/dashboard/pane/mod.rs` > 1500.

### Pitfall 5: Visibility Mismatch

**What goes wrong:** Dashboard.rs can't call methods that were moved to stream_setup.rs.
**Why it happens:** Extracted functions use `pub(super)` but callers are in `dashboard.rs` (grandparent module).
**How to avoid:** Keep public methods on `State` in mod.rs as thin wrappers. `stream_setup.rs` functions are `pub(super)` (visible within `pane/`). State methods remain `pub` and delegate.
**Warning signs:** "method not found" or visibility errors from dashboard.rs.

## Code Examples

### Pattern: Free Function with Field References

```rust
// stream_setup.rs
pub(super) fn apply_ticksize_change(
    content: &mut Content,
    streams: &mut ResolvedStream,
    tick_multiply: TickMultiplier,
    ticker: Option<TickerInfo>,
) -> Option<Effect> {
    if let Some(ticker) = ticker {
        match content {
            Content::Kline { chart: Some(c), .. } => {
                c.change_tick_size(tick_multiply.multiply_with_min_tick_size(ticker));
                c.reset_request_handler();
            }
            // ... other arms
            _ => {}
        }
    }
    // ... depth stream update logic
    // Returns Some(Effect::RefreshStreams) if needed
}
```

### Pattern: Thin Wrapper on State

```rust
// mod.rs - keeps the public API unchanged
pub fn set_content_and_streams(
    &mut self,
    tickers: Vec<TickerInfo>,
    kind: ContentKind,
) -> Vec<StreamKind> {
    stream_setup::build_content_and_streams(
        &mut self.content,
        &mut self.streams,
        &mut self.settings,
        tickers,
        kind,
    )
}
```

### Pattern: Data Insertion Delegation

```rust
// stream_setup.rs
pub(super) fn insert_hist_klines(
    content: &mut Content,
    req_id: Option<uuid::Uuid>,
    timeframe: Timeframe,
    ticker_info: TickerInfo,
    klines: &[Kline],
) {
    // body moved verbatim from State::insert_hist_klines
}

// mod.rs
pub fn insert_hist_klines(
    &mut self,
    req_id: Option<uuid::Uuid>,
    timeframe: Timeframe,
    ticker_info: TickerInfo,
    klines: &[Kline],
) {
    stream_setup::insert_hist_klines(
        &mut self.content, req_id, timeframe, ticker_info, klines,
    )
}
```

## Validation Architecture

### Test Framework

| Property           | Value                                       |
| ------------------ | ------------------------------------------- |
| Framework          | cargo test (Rust built-in)                  |
| Config file        | Cargo.toml workspace                        |
| Quick run command  | `cargo clippy --all-targets -- -D warnings` |
| Full suite command | `mise run lint`                             |

### Phase Requirements -> Test Map

| Req ID  | Behavior                                  | Test Type | Automated Command                                                       | File Exists? |
| ------- | ----------------------------------------- | --------- | ----------------------------------------------------------------------- | ------------ |
| PANE-02 | Stream setup extracted to stream_setup.rs | smoke     | `test -f src/screen/dashboard/pane/stream_setup.rs && cargo check`      | Wave 0       |
| PANE-03 | pane/mod.rs below 1500 LOC                | smoke     | `wc -l src/screen/dashboard/pane/mod.rs` (check < 1500)                 | N/A (shell)  |
| VER-01  | Clippy clean                              | lint      | `cargo clippy --all-targets -- -D warnings`                             | Existing     |
| VER-02  | No behavior change                        | manual    | Open ODB pane, switch tickers, switch basis                             | Manual       |
| VER-03  | No unsafe                                 | grep      | `grep -r "unsafe" src/screen/dashboard/pane/stream_setup.rs` (expect 0) | N/A          |

### Sampling Rate

- **Per task commit:** `cargo clippy --all-targets -- -D warnings`
- **Per wave merge:** `mise run lint`
- **Phase gate:** Full lint + manual ODB pane verification

### Wave 0 Gaps

None -- existing build infrastructure covers all automated checks. Manual ODB verification required.

## Open Questions

1. **Should insert_hist_oi/klines/odb_klines go to stream_setup.rs or content.rs?**
   - What we know: They access only `self.content`. Semantically they are data operations, not stream setup.
   - What's unclear: Phase 4 defined content.rs scope. Adding methods there may conflict with Phase 4's boundary.
   - Recommendation: Put in stream_setup.rs with a `// Data insertion helpers` section comment. These operate on content but are called during the stream data flow pipeline, making stream_setup.rs a reasonable home. Alternative: add them as `Content` methods in content.rs if the planner prefers semantic correctness over phase boundary purity.

## Sources

### Primary (HIGH confidence)

- Direct code analysis of `src/screen/dashboard/pane/mod.rs` (1980 LOC, read in full)
- Direct code analysis of `src/screen/dashboard/pane/content.rs` (465 LOC)
- Direct code analysis of `src/screen/dashboard.rs` call sites for set_content_and_streams
- Phase 4 execution results (pane.rs -> pane/mod.rs + pane/content.rs split)

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - Pure Rust module extraction, no external dependencies
- Architecture: HIGH - Full code read, all field access patterns mapped, LOC counted precisely
- Pitfalls: HIGH - Borrow checker issues identified from actual field access analysis

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable -- internal refactoring, no external API changes)

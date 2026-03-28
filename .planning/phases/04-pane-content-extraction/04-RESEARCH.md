# Phase 4: Pane Content Extraction - Research

**Researched:** 2026-03-27
**Domain:** Rust module extraction (god-module split)
**Confidence:** HIGH

## Summary

Phase 4 extracts the `Content` enum and all its associated methods from `src/screen/dashboard/pane.rs` (2431 LOC) into a new `src/screen/dashboard/pane/content.rs` file. This is a mechanical file-split operation with no behavior changes. The `Content` enum definition starts at line 1880 and its `impl` blocks run through line 2329, totaling approximately 450 LOC. The factory methods (`new_heatmap`, `new_kline`, `placeholder`) and helper methods (`last_tick`, `kind`, `initialized`, `toggle_indicator`, `reorder_indicators`, `change_visual_config`, `studies`, `update_studies`, `chart_kind`) all live in `impl Content` blocks that move together.

The primary complexity is the file-to-directory conversion: `pane.rs` must become `pane/mod.rs`, then `pane/content.rs` is added as a submodule. All external references use `pane::Content` which will be preserved via `pub(crate) use content::Content` in `pane/mod.rs`. The 40+ references in `dashboard.rs` and `layout.rs` will continue to work without changes.

**Primary recommendation:** Convert `pane.rs` to `pane/mod.rs` + `pane/content.rs` in a single atomic step. Move the `Content` enum, all `impl Content` blocks, the `Display` impl, and the `PartialEq` impl. Keep all free functions (`link_group_modal`, `ticksize_modifier`, `basis_modifier`, `by_basis_default`) in `mod.rs` since they produce `Message` types and are view helpers tied to `State`.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- all implementation choices at Claude's discretion.

### Claude's Discretion

All implementation choices are at Claude's discretion -- pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions. Key constraint: pane/mod.rs must re-export Content via `pub(crate) use content::Content` so all external imports remain unchanged.

### Deferred Ideas (OUT OF SCOPE)

None -- infrastructure phase.
</user_constraints>

<phase_requirements>

## Phase Requirements

| ID      | Description                                                                              | Research Support                                                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| PANE-01 | Content type and its factory methods extracted to `pane/content.rs` (~500 LOC reduction) | Full inventory of Content enum (line 1880-1901), impl Content (lines 1903-2329), Display impl (2312-2316), PartialEq impl (2318-2329) = ~450 LOC |
| VER-01  | `cargo clippy -- -D warnings` passes after every phase                                   | No new code -- pure move operation. Clippy passes if imports are correct                                                                         |
| VER-02  | Zero behavior changes -- all existing functionality works identically                    | Pure structural move, no logic changes                                                                                                           |
| VER-03  | No new `unsafe` code introduced                                                          | No unsafe code involved                                                                                                                          |

</phase_requirements>

## Architecture Patterns

### File-to-Directory Conversion Pattern

Converting `pane.rs` to `pane/mod.rs` is the standard Rust pattern for adding submodules to an existing module. The Rust module system treats `foo.rs` and `foo/mod.rs` as equivalent. This means:

1. `git mv src/screen/dashboard/pane.rs src/screen/dashboard/pane/mod.rs` (create dir first)
2. Add `mod content;` to top of `pane/mod.rs`
3. Add `pub(crate) use content::Content;` for re-export
4. Create `pane/content.rs` with extracted code

**Anti-pattern to avoid:** Do NOT use `pane.rs` + `pane/` directory simultaneously. Rust 2024 edition supports only one convention per module. Since this project uses Rust 2024 edition, the directory approach (`pane/mod.rs`) is the correct choice.

### What MOVES to `content.rs`

| Item                                                          | Lines     | Type               | Notes                                                                 |
| ------------------------------------------------------------- | --------- | ------------------ | --------------------------------------------------------------------- |
| `Content` enum definition                                     | 1880-1901 | `pub enum Content` | 6 variants: Starter, Heatmap, Kline, TimeAndSales, Ladder, Comparison |
| `Content::new_heatmap`                                        | 1904-1960 | factory method     | Private (`fn`), called only from `State::set_content_and_streams`     |
| `Content::new_kline`                                          | 1962-2082 | factory method     | Private (`fn`), called only from `State::set_content_and_streams`     |
| `Content::placeholder`                                        | 2084-2135 | factory method     | Private (`fn`), called only from `State::update`                      |
| `Content::last_tick`                                          | 2137-2146 | pub method         | Called from `State::last_tick`                                        |
| `Content::chart_kind`                                         | 2148-2153 | pub method         | Called from `dashboard.rs`                                            |
| `Content::toggle_indicator`                                   | 2155-2195 | pub method         | Called from `State::update`                                           |
| `Content::reorder_indicators`                                 | 2197-2208 | pub method         | Called from `State::update`                                           |
| `Content::change_visual_config`                               | 2210-2234 | pub method         | Called from `dashboard.rs`                                            |
| `Content::studies`                                            | 2236-2251 | pub method         | Called from `dashboard.rs`                                            |
| `Content::update_studies`                                     | 2253-2283 | pub method         | Called from `dashboard.rs`                                            |
| `Content::kind`                                               | 2285-2298 | pub method         | Called from many places                                               |
| `Content::initialized`                                        | 2300-2309 | method             | Private, called from `State`                                          |
| `impl Display for Content`                                    | 2312-2316 | trait impl         |                                                                       |
| `impl PartialEq for Content`                                  | 2318-2329 | trait impl         |                                                                       |
| `#[derive(Default)]` + `#[allow(clippy::large_enum_variant)]` | 1880-1882 | attributes         | Must move with enum                                                   |

### What STAYS in `pane/mod.rs`

| Item                       | Lines     | Reason                                                   |
| -------------------------- | --------- | -------------------------------------------------------- |
| All `use` imports          | 1-45      | Split between mod.rs and content.rs as needed            |
| `Effect` enum              | 47-54     | Part of State's API                                      |
| `Status` enum              | 56-62     | Part of State's API                                      |
| `Action` enum              | 64-71     | Part of State's API                                      |
| `Message` enum             | 73-89     | Part of State's API                                      |
| `Event` enum               | 91-107    | Part of State's API                                      |
| `State` struct + all impls | 109-1878  | The pane state machine -- this is the core of mod.rs     |
| `link_group_modal` fn      | 2331-2371 | Free function producing `Message`, view helper           |
| `ticksize_modifier` fn     | 2373-2397 | Free function producing `Message`, view helper           |
| `basis_modifier` fn        | 2399-2417 | Free function producing `Message`, view helper           |
| `by_basis_default` fn      | 2419-2431 | Generic utility used by `State::set_content_and_streams` |

### Visibility Adjustments Required

The factory methods `new_heatmap`, `new_kline`, and `placeholder` are currently private (`fn` without `pub`). After extraction to `content.rs`, they need to be `pub(super)` so `mod.rs` (which contains `State`) can call them.

Similarly, `initialized()` is private and called from `State::tick()` and `State::view()` -- needs `pub(super)`.

| Method        | Current visibility | Required visibility |
| ------------- | ------------------ | ------------------- |
| `new_heatmap` | private            | `pub(super)`        |
| `new_kline`   | private            | `pub(super)`        |
| `placeholder` | private            | `pub(super)`        |
| `initialized` | private            | `pub(super)`        |

### Import Dependencies for `content.rs`

The `Content` methods use these types that need importing in `content.rs`:

```rust
// From crate (main binary)
use crate::chart::{
    comparison::ComparisonChart,
    heatmap::HeatmapChart,
    kline::KlineChart,
};
use crate::screen::dashboard::panel::{ladder::Ladder, timeandsales::TimeAndSales};
use crate::widget::column_drag;

// From data crate
use data::chart::{
    Basis, ViewConfig,
    indicator::{HeatmapIndicator, KlineIndicator, UiIndicator},
};
use data::layout::pane::{ContentKind, Settings, VisualConfig};

// From exchange crate
use exchange::TickerInfo;

// std
use std::time::Instant;
```

### Cross-Reference Map (Content <-> State)

These are the call sites in `State` (staying in mod.rs) that reference `Content`:

| State method              | Content method called                                                                                           | Pattern                  |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `set_content_and_streams` | `Content::new_heatmap`, `Content::new_kline`, `Content::placeholder` (via `Content::Kline { .. }` construction) | Factory calls            |
| `insert_hist_oi`          | Pattern match on `Content::Kline { chart, .. }`                                                                 | Direct field access      |
| `insert_hist_klines`      | Pattern match on `Content::Kline { .. }`, `Content::Comparison(..)`                                             | Direct field access      |
| `insert_odb_klines`       | Pattern match on `Content::Kline { .. }`                                                                        | Direct field access      |
| `view`                    | Pattern match on all Content variants                                                                           | Read field access        |
| `apply_keyboard_nav`      | Pattern match on `Content::Kline { .. }`                                                                        | Mutable field access     |
| `update`                  | `Content::placeholder`, pattern match on multiple variants                                                      | Factory + field access   |
| `invalidate`              | Pattern match on all Content variants                                                                           | Delegates to inner chart |
| `update_interval`         | Pattern match on all Content variants                                                                           | Read access              |
| `tick`                    | `content.initialized()`, `content.last_tick()`                                                                  | Method calls             |
| `Default for State`       | `Content::Starter`                                                                                              | Variant construction     |

All of these work fine after extraction because:

- `Content` is re-exported as `pub(crate)` from `mod.rs`
- `State` is in `mod.rs` which has `mod content` access (parent module)
- All public methods and `pub(super)` methods are accessible

### No Circular Dependency Risk

`content.rs` depends on chart types and data types only. It does NOT reference `State`, `Message`, `Event`, `Effect`, or `Action`. The dependency flows one way: `mod.rs (State) -> content.rs (Content)`. No risk of circular imports.

## Don't Hand-Roll

| Problem                      | Don't Build            | Use Instead                    | Why                              |
| ---------------------------- | ---------------------- | ------------------------------ | -------------------------------- |
| File-to-directory conversion | Manual copy-paste      | `mkdir` + `git mv`             | Preserves git history            |
| Import resolution            | Manual import guessing | `cargo clippy` after each step | Compiler catches missing imports |

## Common Pitfalls

### Pitfall 1: Forgetting `pub(super)` on Previously-Private Methods

**What goes wrong:** `State` methods in `mod.rs` can't call `Content::new_kline()` etc.
**Why it happens:** These methods were private because they were in the same file. Now they're in a submodule.
**How to avoid:** Change `fn new_heatmap`, `fn new_kline`, `fn placeholder`, `fn initialized` to `pub(super) fn`.
**Warning signs:** Compilation error "method is private" on `Content::new_kline`.

### Pitfall 2: Missing the `#[allow(clippy::large_enum_variant)]` Attribute

**What goes wrong:** Clippy warns about large enum variant size difference.
**Why it happens:** The attribute is on the `Content` enum and must move with it.
**How to avoid:** Move the `#[derive(Default)]` and `#[allow]` attributes together with the enum.
**Warning signs:** New clippy warning about `Content` enum.

### Pitfall 3: Duplicate Imports

**What goes wrong:** Both `mod.rs` and `content.rs` import the same types, or `mod.rs` has unused imports after extraction.
**Why it happens:** The original `use` block at the top of `pane.rs` serves both `State` and `Content`.
**How to avoid:** After extraction, run `cargo clippy` to detect unused imports in `mod.rs` and missing imports in `content.rs`.
**Warning signs:** Clippy `unused_imports` warnings.

### Pitfall 4: Forgetting `mod content;` Declaration

**What goes wrong:** Rust doesn't know about the new file.
**Why it happens:** Creating the file without declaring it as a module.
**How to avoid:** Add `mod content;` near the top of `mod.rs`, before the `pub(crate) use content::Content;` re-export.

### Pitfall 5: Git History Loss

**What goes wrong:** `git log --follow content.rs` shows no history.
**Why it happens:** Creating a new file instead of using `git mv` + edit.
**How to avoid:** Use `git mv pane.rs pane/mod.rs` for the rename, then create `content.rs` as a new file. The main history stays with `mod.rs` (the larger file). `content.rs` is genuinely new -- git can't track "part of a file moved" anyway, but keeping `mod.rs` as the rename target preserves the bulk of history.

## Validation Architecture

### Test Framework

| Property           | Value                                                   |
| ------------------ | ------------------------------------------------------- |
| Framework          | cargo clippy + cargo build (no test suite for GUI code) |
| Config file        | `clippy.toml`, `rustfmt.toml`                           |
| Quick run command  | `cargo clippy --all-targets -- -D warnings`             |
| Full suite command | `mise run lint`                                         |

### Phase Requirements -> Test Map

| Req ID  | Behavior                        | Test Type      | Automated Command                                       | File Exists? |
| ------- | ------------------------------- | -------------- | ------------------------------------------------------- | ------------ |
| PANE-01 | Content extracted to content.rs | structural     | `test -f src/screen/dashboard/pane/content.rs`          | Wave 0       |
| VER-01  | Clippy clean                    | lint           | `cargo clippy --all-targets -- -D warnings`             | Existing     |
| VER-02  | No behavior change              | build + manual | `cargo build` + launch app                              | Existing     |
| VER-03  | No unsafe code                  | grep           | `grep -r "unsafe" src/screen/dashboard/pane/content.rs` | Wave 0       |

### Sampling Rate

- **Per task commit:** `cargo clippy --all-targets -- -D warnings`
- **Per wave merge:** `mise run lint`
- **Phase gate:** Full lint green + file existence check

### Wave 0 Gaps

None -- existing lint infrastructure covers all phase requirements. No new test files needed.

## Code Examples

### Re-export Pattern in mod.rs

```rust
// src/screen/dashboard/pane/mod.rs (top of file, after use statements)
mod content;
pub(crate) use content::Content;
```

### Visibility Change Pattern in content.rs

```rust
// Previously private in pane.rs, now pub(super) in content.rs
impl Content {
    pub(super) fn new_heatmap(
        current_content: &Content,
        ticker_info: TickerInfo,
        settings: &Settings,
        tick_size: f32,
    ) -> Self {
        // ... unchanged body
    }

    pub(super) fn new_kline(
        content_kind: ContentKind,
        current_content: &Content,
        ticker_info: TickerInfo,
        settings: &Settings,
        tick_size: f32,
    ) -> Self {
        // ... unchanged body
    }

    pub(super) fn placeholder(kind: ContentKind) -> Self {
        // ... unchanged body
    }

    pub(super) fn initialized(&self) -> bool {
        // ... unchanged body
    }
}
```

## Project Constraints (from CLAUDE.md)

- **No regressions**: Must compile clean (`cargo clippy -- -D warnings`) after every phase
- **No behavior change**: All existing functionality must work identically
- **Incremental**: Each phase is independently shippable
- **Rust edition**: 2024, toolchain 1.93.1
- **Max line width**: 100 characters (rustfmt.toml)
- **FILE-SIZE-OK comments**: Original `pane.rs` has one at line 1 -- keep it in `mod.rs` (upstream file convention)
- **Fork-specific comments**: Preserve `// GitHub Issue:` and `// NOTE(fork):` comments in moved code
- **Lint command**: `mise run lint` = `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings`

## Open Questions

None -- this is a fully understood mechanical extraction with no ambiguity.

## Sources

### Primary (HIGH confidence)

- Direct code analysis of `src/screen/dashboard/pane.rs` (2431 lines read in full)
- Grep analysis of all `Content` references across `src/` directory (40+ call sites identified)
- `dashboard.rs` module structure verified (line 3: `pub mod pane;`)

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - pure Rust module system, no libraries involved
- Architecture: HIGH - full file read, all cross-references mapped
- Pitfalls: HIGH - standard file-split operation, pitfalls are well-known

**Research date:** 2026-03-27
**Valid until:** Indefinite (Rust module system is stable)

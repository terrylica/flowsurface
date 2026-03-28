# Phase 2: Must-Use Safety Net - Research

**Researched:** 2026-03-27
**Domain:** Rust `#[must_use]` attribute on return types in iced Elm-architecture
**Confidence:** HIGH

## Summary

This phase adds `#[must_use]` annotations to `Action`, `Effect`, and `Task`-returning methods in `pane.rs` and `dashboard.rs` so the compiler warns when return values are silently dropped. This is a safety net before phases 4-7 move code between modules, where dropped returns would silently break event dispatch.

**Critical finding:** `#[must_use]` on an enum does NOT propagate through `Option<T>`. Since most methods return `Option<Effect>` or `Option<Action>` (not bare types), we need `#[must_use]` on both the enums AND the functions. Annotating only the enums would miss the majority of call sites.

**Primary recommendation:** Apply `#[must_use]` to all four target enums (`pane::Effect`, `pane::Action`, `chart::Action`, `panel::Action`) and to every function returning `Option<Effect>` or `Option<Action>` in pane.rs, dashboard.rs, and related chart/panel modules. Fix any resulting warnings (one known: `invalidate_all_panes` drops `Option<Action>`).

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- infrastructure phase, all choices at Claude's discretion.

### Claude's Discretion

All implementation choices are at Claude's discretion -- pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

### Deferred Ideas (OUT OF SCOPE)

None.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID      | Description                                                                                          | Research Support                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| QUAL-02 | `#[must_use]` added to Task/Effect return types in pane.rs and dashboard.rs before any code is moved | Inventory of all target types and functions identified; `Option<T>` propagation behavior verified experimentally |
| VER-01  | `cargo clippy -- -D warnings` passes after every phase                                               | Current clippy passes clean; one known drop site (`let _ = state.invalidate()`) is intentional and safe          |
| VER-02  | Zero behavior changes -- all existing functionality works identically                                | `#[must_use]` is purely a compile-time lint, zero runtime impact                                                 |
| VER-03  | No new `unsafe` code introduced                                                                      | No unsafe needed                                                                                                 |

</phase_requirements>

## Standard Stack

No new dependencies. This phase uses only built-in Rust attributes.

| Tool           | Version          | Purpose                                    | Why Standard                          |
| -------------- | ---------------- | ------------------------------------------ | ------------------------------------- |
| `#[must_use]`  | Rust built-in    | Compile-time lint for unused return values | Part of the language; no crate needed |
| `cargo clippy` | 1.93.1 toolchain | Verification that no warnings introduced   | Already required by VER-01            |

## Architecture Patterns

### Pattern 1: `#[must_use]` on Enums

**What:** Annotate the enum definition directly.
**When:** Type is returned bare (not wrapped in `Option`).
**Limitation:** Does NOT propagate through `Option<T>` -- verified experimentally with rustc 2024 edition.

```rust
#[must_use = "Effect must be handled by the caller"]
pub enum Effect {
    RefreshStreams,
    RequestFetch(Vec<FetchSpec>),
    SwitchTickersInGroup(TickerInfo),
    FocusWidget(iced::widget::Id),
}
```

### Pattern 2: `#[must_use]` on Functions Returning `Option<T>`

**What:** Annotate the function itself when it returns `Option<MustUseType>`.
**When:** Most pane/chart methods return `Option<Effect>` or `Option<Action>`.
**Why needed:** `Option<T>` does not inherit `#[must_use]` from `T`.

```rust
#[must_use = "returned Effect must be dispatched"]
pub fn update(&mut self, msg: Event) -> Option<Effect> { ... }

#[must_use = "returned Action must be dispatched"]
pub fn invalidate(&mut self, now: Instant) -> Option<Action> { ... }
```

### Pattern 3: Intentional Drops with `let _ =`

**What:** Explicit `let _ = expr;` suppresses `#[must_use]` warnings.
**When:** The caller genuinely does not need the result.
**Existing usage:** `dashboard.rs:1315` -- `invalidate_all_panes` intentionally drops `Option<Action>` from each pane's `invalidate()` because it only needs the side effect (geometry cache clear), not the action.

```rust
// This is already correct -- intentional drop, no change needed
let _ = state.invalidate(Instant::now());
```

### Anti-Patterns to Avoid

- **Annotating only enums:** Misses `Option<T>` call sites (the majority).
- **Annotating `Option` itself:** Cannot add `#[must_use]` to std's `Option` -- must annotate at function level.
- **Over-annotating private helpers:** Only annotate public/pub(crate) functions that cross module boundaries. Private functions called in a single place add noise.

## Target Inventory

### Enums to Annotate with `#[must_use]`

| Enum                    | File                                       | Variants                                                              |
| ----------------------- | ------------------------------------------ | --------------------------------------------------------------------- |
| `pane::Effect`          | `src/screen/dashboard/pane.rs:48`          | RefreshStreams, RequestFetch, SwitchTickersInGroup, FocusWidget       |
| `pane::Action`          | `src/screen/dashboard/pane.rs:63`          | Chart, Panel, ResolveStreams, ResolveContent                          |
| `chart::Action`         | `src/chart.rs:60`                          | ErrorOccurred, RequestFetch                                           |
| `panel::Action`         | `src/screen/dashboard/panel.rs:17`         | (empty enum -- annotate for forward safety)                           |
| `comparison::Action`    | `src/chart/comparison.rs:16`               | SeriesColorChanged, SeriesNameChanged, RemoveSeries, OpenSeriesEditor |
| `tickers_table::Action` | `src/screen/dashboard/tickers_table.rs:56` | TickerSelected, SyncToAllPanes, ErrorOccurred, Fetch, FocusWidget     |
| `sidebar::Action`       | `src/screen/dashboard/sidebar.rs:30`       | TickerSelected, SyncToAllPanes, ErrorOccurred                         |

Note: `iced::Task` already has `#[must_use]` (verified in iced_runtime 0.14.0 source). No action needed for Task-returning methods.

### Functions Returning `Option<Effect>` (need `#[must_use]`)

| Function                         | File:Line      | Return Type                                       |
| -------------------------------- | -------------- | ------------------------------------------------- |
| `State::update()`                | `pane.rs:1139` | `Option<Effect>`                                  |
| `State::show_modal_with_focus()` | `pane.rs:1753` | `Option<Effect>` (private -- annotate for safety) |

### Functions Returning `Option<Action>` (need `#[must_use]`)

| Function                                    | File:Line              | Return Type                            |
| ------------------------------------------- | ---------------------- | -------------------------------------- |
| `State::invalidate()`                       | `pane.rs:1776`         | `Option<Action>`                       |
| `State::tick()`                             | `pane.rs:1816`         | `Option<Action>`                       |
| `KlineChart::missing_data_task()`           | `kline/mod.rs:655`     | `Option<chart::Action>` (private)      |
| `KlineChart::set_basis()`                   | `kline/mod.rs:916`     | `Option<chart::Action>`                |
| `KlineChart::invalidate()`                  | `kline/mod.rs:1138`    | `Option<chart::Action>`                |
| `ComparisonChart::update()`                 | `comparison.rs:92`     | `Option<comparison::Action>`           |
| `ComparisonChart::open_editor_for_ticker()` | `comparison.rs:388`    | `Option<comparison::Action>` (private) |
| `Panel::invalidate()`                       | `panel.rs:24` (trait)  | `Option<panel::Action>`                |
| `TickersTable::update()`                    | `tickers_table.rs:159` | `Option<tickers_table::Action>`        |

### Task-Returning Methods (already covered by iced's `#[must_use]`)

| Function                       | File:Line           | Notes                     |
| ------------------------------ | ------------------- | ------------------------- |
| `Dashboard::load_layout()`     | `dashboard.rs:162`  | Already warned if dropped |
| `Dashboard::focus_pane()`      | `dashboard.rs:601`  | Already warned if dropped |
| `Dashboard::split_pane()`      | `dashboard.rs:609`  | Already warned if dropped |
| `Dashboard::popout_pane()`     | `dashboard.rs:623`  | Already warned if dropped |
| `Dashboard::merge_pane()`      | `dashboard.rs:648`  | Already warned if dropped |
| `Dashboard::refresh_streams()` | `dashboard.rs:1482` | Already warned if dropped |

### Known Intentional Drops (should remain as `let _ =`)

| Location            | Expression                                 | Why Intentional                                                                                                                           |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `dashboard.rs:1315` | `let _ = state.invalidate(Instant::now())` | `invalidate_all_panes` only needs side effect (cache clear); Actions would need pane context to dispatch, which `for_each` cannot provide |

## Don't Hand-Roll

| Problem                        | Don't Build                  | Use Instead             | Why                                                  |
| ------------------------------ | ---------------------------- | ----------------------- | ---------------------------------------------------- |
| Return-value-dropped detection | Custom lint or clippy plugin | `#[must_use]` attribute | Built into rustc, zero overhead, standard Rust idiom |

## Common Pitfalls

### Pitfall 1: Option<T> Does NOT Inherit #[must_use]

**What goes wrong:** Developer annotates only the enum, assumes `Option<Effect>` calls are covered.
**Why it happens:** Intuition says "Option of must-use should be must-use," but Rust does not propagate this.
**How to avoid:** Always annotate the function when it returns `Option<MustUseType>`.
**Warning signs:** No new warnings appear after annotating enums (because all call sites return Option).

### Pitfall 2: Clippy vs Rustc Warning Interaction

**What goes wrong:** `cargo clippy -- -D warnings` may surface `unused_must_use` warnings from the annotation changes that break the build.
**Why it happens:** Existing code that intentionally drops values (like `invalidate_all_panes`) would now fail clippy.
**How to avoid:** Audit all call sites BEFORE adding annotations. Ensure existing `let _ =` patterns are in place.
**Warning signs:** Clippy fails after adding annotations -- check for missing `let _ =` at intentional drop sites.

### Pitfall 3: Trait Method Annotations

**What goes wrong:** Adding `#[must_use]` to a trait method definition does not always propagate to implementations.
**Why it happens:** Trait method `#[must_use]` does work in Rust (since 1.67), but only if called through the trait. Direct calls to impl methods need their own annotation.
**How to avoid:** For `Panel::invalidate()` trait method, annotate at the trait level -- it covers all trait-dispatch call sites.

## Validation Architecture

### Test Framework

| Property           | Value                                       |
| ------------------ | ------------------------------------------- |
| Framework          | cargo clippy (lint-based verification)      |
| Config file        | `clippy.toml` (existing)                    |
| Quick run command  | `cargo clippy --all-targets -- -D warnings` |
| Full suite command | `cargo clippy --all-targets -- -D warnings` |

### Phase Requirements to Test Map

| Req ID  | Behavior                                         | Test Type           | Automated Command                                             | File Exists? |
| ------- | ------------------------------------------------ | ------------------- | ------------------------------------------------------------- | ------------ |
| QUAL-02 | `#[must_use]` on Effect/Action types and methods | compile-time lint   | `cargo clippy --all-targets -- -D warnings`                   | N/A (lint)   |
| VER-01  | Clippy passes clean                              | compile-time lint   | `cargo clippy --all-targets -- -D warnings`                   | N/A (lint)   |
| VER-02  | Zero behavior changes                            | manual verification | `mise run run` (launch and verify ODB panes work)             | N/A (manual) |
| VER-03  | No new unsafe                                    | grep audit          | `grep -rn "unsafe" src/ --include="*.rs"` (diff before/after) | N/A (grep)   |

### Sampling Rate

- **Per task commit:** `cargo clippy --all-targets -- -D warnings`
- **Per wave merge:** Same (single-wave phase)
- **Phase gate:** Full clippy green before `/gsd:verify-work`

### Wave 0 Gaps

None -- existing clippy infrastructure covers all phase requirements. No new test files needed.

## Code Examples

### Annotating an Enum

```rust
// Source: Rust Reference - #[must_use] attribute
#[must_use = "Effect must be handled by the caller"]
#[derive(Debug, Clone)]
pub enum Effect {
    RefreshStreams,
    RequestFetch(Vec<FetchSpec>),
    SwitchTickersInGroup(TickerInfo),
    FocusWidget(iced::widget::Id),
}
```

### Annotating a Function Returning Option<Effect>

```rust
// Must annotate the function because Option<T> doesn't inherit #[must_use] from T
#[must_use = "returned Effect must be dispatched to dashboard"]
pub fn update(&mut self, msg: Event) -> Option<Effect> {
    // ...existing code unchanged...
}
```

### Annotating a Trait Method

```rust
pub trait Panel: canvas::Program<Message> {
    fn scroll(&mut self, scroll: f32);
    fn reset_scroll(&mut self);

    #[must_use = "returned Action must be handled"]
    fn invalidate(&mut self, now: Option<Instant>) -> Option<Action>;

    fn is_empty(&self) -> bool;
}
```

## Sources

### Primary (HIGH confidence)

- Rust Reference: `#[must_use]` attribute behavior -- verified experimentally with rustc 2024 edition
- iced_runtime 0.14.0 source (`task.rs:23`) -- `Task` already has `#[must_use]`
- Codebase audit -- complete inventory of all Effect/Action types and their call sites

### Verification Experiments

- `/tmp/must_use_test2.rs` -- confirmed `#[must_use]` on enum warns for bare returns
- `/tmp/must_use_test4.rs` -- confirmed `Option<#[must_use] T>` does NOT warn (critical finding)
- `/tmp/must_use_test3.rs` -- confirmed `#[must_use]` on function warns for `Option<T>` returns

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - built-in Rust attribute, no dependencies
- Architecture: HIGH - experimentally verified `Option<T>` propagation behavior
- Pitfalls: HIGH - Option non-propagation verified; existing drop sites audited
- Target inventory: HIGH - grep-based audit of all return types

**Research date:** 2026-03-27
**Valid until:** Indefinite (Rust attribute semantics are stable)

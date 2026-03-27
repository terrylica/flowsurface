# Architecture Patterns

**Domain:** Rust iced 0.14 desktop application refactoring
**Researched:** 2026-03-26

## Recommended Architecture

### Guiding Principle: Organize by Feature, Not by Layer

The iced maintainers explicitly advise against splitting State/Message/Update/View into separate files (discussion #1572). The Elm Architecture philosophy treats these as tightly coupled parts of the same abstraction. Splitting by layer creates boundary ambiguity (where does a helper used by both `update` and `view` go?).

**Instead: organize modules around types and features.** A `KlineChart` module contains its state, messages, update logic, and view code together. Sub-features (bar selection, crosshair, ODB processing) extract into submodules that the parent orchestrates.

This is critical context for the refactoring: the goal is not to separate concerns by architectural layer, but to split god modules along **feature boundaries** within the same type.

### Current Architecture (What Exists)

```
main.rs (Flowsurface)
  |
  +-- update() dispatches Message variants
  |     |
  |     +-- Dashboard::update()          [1906 LOC - dashboard.rs]
  |           |
  |           +-- pane::State::update()   [2425 LOC - pane.rs]
  |                 |
  |                 +-- chart::update()   [chart.rs - 425 LOC, clean]
  |                 +-- KlineChart methods [2388 LOC - kline/mod.rs]
  |
  +-- view() renders Element tree
        |
        +-- Dashboard::view()
              |
              +-- pane::State::view()
                    |
                    +-- chart::view() -> Canvas<KlineChart>
                          |
                          +-- KlineChart::draw()  [canvas::Program impl]
```

### Target Architecture (Where to Go)

```
main.rs (Flowsurface) ............... entry, message routing
  |
  +-- screen/dashboard/
  |     +-- mod.rs .................. Dashboard struct, pane grid, stream distribution
  |     +-- pane/
  |     |     +-- mod.rs ............ State struct, update(), view() - SLIM orchestrator
  |     |     +-- content.rs ........ Content enum, new_kline/new_heatmap factories
  |     |     +-- stream_setup.rs ... resolve_content(), set_content_and_streams()
  |     |     +-- settings_ui.rs .... view() helpers for settings panel, title bar
  |     |
  +-- chart/
        +-- mod.rs .................. Chart trait, view(), update() - UNCHANGED
        +-- kline/
        |     +-- mod.rs ............ KlineChart struct, new(), Chart impl - SLIM
        |     +-- data_ops.rs ....... insert_hist_klines, insert_open_interest, insert_trades
        |     +-- odb_lifecycle.rs .. ODB processor, gap-fill, reconciliation, sentinel
        |     +-- program.rs ........ canvas::Program impl (update + draw)
        |     +-- rendering.rs ...... draw helpers (existing, already extracted)
        |     +-- crosshair.rs ...... crosshair tooltip (existing, already extracted)
        |     +-- bar_selection.rs .. bar selection (existing, already extracted)
        |     +-- odb_core.rs ....... ODB validation (existing, already extracted)
        |
        +-- interaction.rs .......... canvas interaction (existing, clean)
        +-- view_state.rs ........... ViewState (existing, clean)
```

## Component Boundaries

### Boundary 1: KlineChart (kline/mod.rs -> split into 4)

The 2388-LOC `kline/mod.rs` contains three distinct concerns sharing one struct:

| Component                  | Responsibility                                                                                            | Communicates With                                      |
| -------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `KlineChart` core (mod.rs) | Struct definition, `new()`, `Chart` trait impl, `PlotConstants`                                           | All submodules                                         |
| `data_ops`                 | Historical kline insertion, open interest, trade fetching                                                 | `data_source` (PlotData), indicators                   |
| `odb_lifecycle`            | ODB processor management, gap-fill fence, SSE reconciliation, sentinel audit, trade ring buffer, watchdog | `data_source`, `odb_processor`, dashboard (via Action) |
| `program`                  | `canvas::Program` impl (`update` + `draw`)                                                                | `ViewState`, `bar_selection`, `rendering`, `crosshair` |

**Why this split works:** The struct fields naturally partition into groups:

- **Core fields** (7): `chart`, `data_source`, `raw_trades`, `indicators`, `kind`, `request_handler`, `study_configurator`, `kline_config`
- **ODB-specific fields** (20): `odb_processor`, `next_agg_id`, `gap_fill_fence_agg_id`, `buffered_ch_klines`, `ws_trade_ring`, `sse_reset_fence_agg_id`, sentinel fields, watchdog fields, telemetry counters
- **Interaction state** (1 RefCell): `bar_selection`

The struct itself stays in `mod.rs`. Methods move to `impl KlineChart` blocks in submodule files. Rust allows `impl` blocks across files within the same module through `mod` + file adjacency.

**iced constraint:** `canvas::Program` takes `&self` (immutable). The `draw()` and `update()` methods use `RefCell` for interior mutability. These two methods must stay together or at least in the same file, because they share the `&self` borrow pattern and `RefCell` discipline. Putting `draw` in one file and `update` in another risks subtle borrow conflicts when reviewers cannot see both sides.

### Boundary 2: Pane State (pane.rs -> split into 3-4)

The 2425-LOC `pane.rs` mixes:

| Component             | Responsibility                                                                   | Communicates With                    |
| --------------------- | -------------------------------------------------------------------------------- | ------------------------------------ |
| `State` core (mod.rs) | Struct, `update()`, message dispatch                                             | Content, Dashboard                   |
| `content.rs`          | `Content` enum, factory methods (`new_kline`, `new_heatmap`), indicator toggling | KlineChart, HeatmapChart, data crate |
| `stream_setup.rs`     | `set_content_and_streams()`, `resolve_content()`, `by_basis_default()`           | Exchange crate (StreamKind), Content |
| `settings_ui.rs`      | Title bar view, settings panel, modal views                                      | Style, Modals                        |

**Why this split works:** The `Content` enum (lines 1877-1895) with its 7 factory methods and 15+ helper methods is an independent type that happens to live inside pane.rs. Stream setup logic is pure routing (ticker + content kind -> Vec<StreamKind>) with no rendering concerns. The `view()` function at 560 lines is almost entirely UI layout code.

### Boundary 3: Dashboard (dashboard.rs - 1906 LOC)

Dashboard is borderline for splitting. Its `update()` method is a single large match on `Message` variants. Possible extraction:

| Component                | Responsibility                                                         |
| ------------------------ | ---------------------------------------------------------------------- |
| `stream_distribution.rs` | `distribute_fetched_data()`, `update_latest_klines()` (lines 999-1365) |
| `stream_resolution.rs`   | `resolve_streams()`, `refresh_streams()` (lines 1367-1530)             |

**Caution:** Dashboard's `update()` returns `(Task<Message>, Option<Event>)`. Extracting methods that return Tasks requires careful handling -- the methods need `&mut self` access to `self.panes` and `self.popout`. This is feasible via `impl Dashboard` blocks in separate files but adds cognitive overhead for a file that is upstream-controlled.

**Recommendation:** Defer dashboard splitting. The file is marked "upstream structure" and splitting it increases upstream merge friction. Focus on pane.rs and kline/mod.rs which are fork-divergent.

## Patterns to Follow

### Pattern 1: Impl-Block-Per-File (Primary Split Strategy)

**What:** Keep the struct definition in `mod.rs`, distribute `impl` blocks across sibling files in the same directory.

**When:** Struct has 20+ methods that cluster into 2-4 coherent groups.

**Example:**

```rust
// src/chart/kline/mod.rs
pub struct KlineChart {
    // all fields here -- single source of truth
    chart: ViewState,
    data_source: PlotData<KlineDataPoint>,
    // ... all other fields
}

mod data_ops;      // impl KlineChart { fn insert_hist_klines ... }
mod odb_lifecycle;  // impl KlineChart { fn finalize_gap_fill ... }
mod program;        // impl canvas::Program<Message> for KlineChart { ... }

// Chart trait impl stays here (small, 80 lines)
impl Chart for KlineChart { ... }
```

**Why this works in Rust:** `impl` blocks for a type can appear in any module that has access to the type. Since submodules of `kline/` automatically have access to `KlineChart` (it is `pub` in the parent), each submodule file can contain an `impl KlineChart` block. No trait gymnastics needed.

**Visibility rule:** Fields accessed by submodules need `pub(super)` or `pub(crate)`. Currently many fields are private (default). The split requires relaxing visibility to `pub(crate)` for fields used across submodule files. This is acceptable -- all files are within the same crate.

### Pattern 2: Enum-Variant-Per-File (Content Extraction)

**What:** Move each variant's construction and method logic to its own file, keep the enum definition in the parent.

**When:** Enum has 5+ variants each with substantial construction logic.

**Example:**

```rust
// src/screen/dashboard/pane/content.rs
pub enum Content {
    Starter,
    Heatmap { chart: Option<HeatmapChart>, ... },
    Kline { chart: Option<KlineChart>, ... },
    // ...
}

impl Content {
    pub fn new_kline(...) -> Self { ... }
    pub fn new_heatmap(...) -> Self { ... }
    pub fn kind(&self) -> ContentKind { ... }
    pub fn toggle_indicator(&mut self, ...) { ... }
    // all Content methods move here
}
```

### Pattern 3: canvas::Program Interior Mutability Discipline

**What:** All mutable state accessed in `canvas::Program::update(&self)` and `draw(&self)` must use `RefCell<T>`. Never hold borrows across method boundaries.

**When:** Any canvas-rendered chart.

**Critical rule for the split:** The `canvas::Program` impl (`update` + `draw`) should live in a single file (`program.rs`) because:

1. Both methods take `&self` -- they share the same borrow context
2. `RefCell` borrow discipline requires seeing all borrow sites together
3. `draw()` reads from `RefCell`s that `update()` mutates
4. Splitting update/draw across files makes it easy to introduce `BorrowMutError` panics at runtime

### Pattern 4: Action/Effect Return Type for Cross-Module Communication

**What:** Submodule methods return `Option<Action>` or `Option<Effect>` rather than directly modifying parent state.

**When:** A method in a submodule needs to trigger behavior in the parent (e.g., ODB lifecycle detecting a gap needs to request a fetch).

**Example:**

```rust
// odb_lifecycle.rs
impl KlineChart {
    pub(crate) fn audit_bar_continuity(&mut self) -> Option<Action> {
        // ... detect gap ...
        Some(Action::RequestFetch(vec![fetch_spec]))
    }
}

// mod.rs calls it in invalidate()
if let Some(action) = self.audit_bar_continuity() {
    return Some(action);
}
```

This pattern already exists in the codebase (`Action::RequestFetch`, `pane::Effect::RefreshStreams`). The split preserves it -- submodule methods return Actions, parent orchestrates.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Layer-Based Module Splitting

**What:** Separating state.rs / messages.rs / update.rs / view.rs for a single component.

**Why bad:** The iced maintainers explicitly warn against this (discussion #1572). State, messages, update, and view are the same abstraction. Separating them creates boundary confusion and fights the Elm Architecture.

**Instead:** Split by feature within the type. `odb_lifecycle.rs` contains ODB-related state manipulation AND any ODB-specific rendering helpers. `data_ops.rs` contains data insertion AND any data-specific validation.

### Anti-Pattern 2: Trait Extraction for Internal Splits

**What:** Creating a `OdbLifecycle` trait so you can put ODB methods in a separate file via `impl OdbLifecycle for KlineChart`.

**Why bad:** Adds indirection without benefit. Traits are for polymorphism (multiple implementors) or external extension points. Internal method grouping is better served by `impl` blocks in submodule files.

**Instead:** Use bare `impl KlineChart` blocks in each submodule file. No trait needed.

### Anti-Pattern 3: Splitting canvas::Program::draw and canvas::Program::update

**What:** Putting `draw()` in `rendering.rs` and `update()` in `interaction.rs`.

**Why bad:** Both take `&self`. Both access `RefCell` fields. Runtime `BorrowMutError` panics become invisible when you cannot see both borrow sites. The iced `canvas::Program` trait requires both methods in one `impl` block anyway (it is a single trait).

**Instead:** Keep the entire `impl canvas::Program<Message> for KlineChart` in one file (`program.rs`). Delegate heavy rendering to helper functions in `rendering.rs` (already done), but the trait impl stays unified.

## Suggested Split Order (Build Dependencies)

The splits have a dependency chain based on which modules import from which:

```
Phase 1: Content extraction from pane.rs
    |  (no other module depends on Content being in pane.rs)
    |
Phase 2: KlineChart data_ops extraction
    |  (data_ops methods are called by pane.rs and dashboard.rs)
    |  (after Content extraction, pane.rs is smaller and easier to verify)
    |
Phase 3: KlineChart odb_lifecycle extraction
    |  (depends on data_ops being stable -- odb_lifecycle calls insert methods)
    |
Phase 4: KlineChart program.rs extraction
    |  (canvas::Program is the most complex piece -- draw/update/interaction)
    |  (do last because it has the most subtle RefCell borrow constraints)
    |
Phase 5: Pane stream_setup and settings_ui extraction
    |  (depends on Content being in its own file)
    |
[Optional] Phase 6: Dashboard stream distribution extraction
    (defer if upstream merge friction is a concern)
```

**Rationale for this order:**

1. Content extraction is the safest -- it is a self-contained enum with factory methods, zero risk of breaking message dispatch
2. data_ops is called by external code (dashboard distributes klines to charts) -- extract early so the API surface is clear
3. odb_lifecycle is fork-specific, complex, and self-contained -- extracting it makes kline/mod.rs readable
4. program.rs is the riskiest split due to RefCell -- do it last when the struct is well-understood from previous splits
5. Pane UI extraction is low-risk but depends on Content being separate

### Compile Verification at Each Phase

After each extraction:

```bash
cargo clippy -- -D warnings
```

The Rust compiler enforces that all moved methods still resolve. If a method in `data_ops.rs` references a private field, compilation fails immediately. This makes the refactoring mechanically safe -- the risk is logic errors from RefCell misuse, not missing imports.

## iced-Specific Constraints on Module Boundaries

### Constraint 1: Message Enum Must Be Unified Per Level

Each level of the Elm hierarchy has ONE `Message` enum:

- `Flowsurface::Message` (main.rs) -- top level
- `dashboard::Message` (dashboard.rs) -- dashboard level
- `pane::Message` + `pane::Event` (pane.rs) -- pane level
- `chart::Message` (interaction.rs) -- chart level

You **cannot** split these into per-file message enums without fundamentally changing the dispatch architecture. The Message enum stays where it is. Submodule methods receive specific data, not messages.

### Constraint 2: canvas::Program Is &self Only

The `canvas::Program::update()` method takes `&self`, not `&mut self`. All mutation goes through `RefCell<T>`. This means:

- `draw()` and `update()` compete for borrows
- `RefCell::borrow()` in draw and `RefCell::borrow_mut()` in update can panic if overlapping
- In practice iced serializes draw/update calls, but the compiler does not enforce this -- runtime panics are the failure mode

**Implication for splits:** Any file that touches `RefCell` fields must be reviewed holistically. Preferably, only `program.rs` and `bar_selection.rs` (already extracted) touch RefCells.

### Constraint 3: Task<Message> Return Type Threads Through Everything

`Dashboard::update()` returns `Task<Message>`. `Task` is a lazy async effect that the iced runtime executes. This means:

- Methods returning `Task` must have access to async-capable constructors (`Task::perform`, `Task::done`)
- The `Message` type parameter must match the parent's `Message` enum
- Submodule methods that need async effects return `Option<Action>` which the parent converts to `Task`

### Constraint 4: Element Lifetime Borrows from &self

`view(&self) -> Element<'_, Message>` borrows from `self`. This means:

- The `view` method and any UI helper functions need `&self` (or `&Content`, `&Settings`)
- Extracted view helpers must take references to the specific data they need, not `&mut self`
- UI helpers can live in separate files freely -- they are pure functions of references

## Scalability Considerations

| Concern         | Current (2.4K LOC files)                                    | After Split (400-600 LOC files)                                   | Future Growth                                              |
| --------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| New ODB feature | Touch kline/mod.rs (2388 LOC), hunt for right section       | Touch odb_lifecycle.rs (600 LOC), clear location                  | New ODB submodule if needed                                |
| New indicator   | Touch 6+ files (enum, display, factory, renderer, settings) | Same ceremony (indicator pattern is cross-cutting, not file-size) | Consider indicator registry pattern later                  |
| Upstream merge  | Merge 2 god files with heavy fork changes                   | Merge smaller files, conflicts more localized                     | Stream setup / content split reduces upstream diff surface |
| Code review     | 2400 LOC diffs are unreadable                               | 400-600 LOC files, reviewable in one screen                       | Each submodule independently reviewable                    |
| Compile time    | No change from splits (same crate)                          | No change                                                         | Crate splits would help but are out of scope               |

## Sources

- [iced-rs discussion #1572: Module splitting advice](https://github.com/iced-rs/iced/discussions/1572) -- maintainer hecrj's explicit guidance against layer-based splitting
- [iced discourse: Reusable components and app design](https://discourse.iced.rs/t/questions-about-reusable-components-and-app-design-pattern/546) -- Action/Effect pattern for component communication
- [iced canvas::Program trait docs](https://docs.rs/iced/latest/iced/widget/canvas/trait.Program.html) -- `&self` constraint on update/draw
- [iced architecture guide](https://book.iced.rs/architecture.html) -- Elm Architecture overview
- Codebase analysis: `src/chart/kline/mod.rs` (2388 LOC), `src/screen/dashboard/pane.rs` (2425 LOC), `src/screen/dashboard.rs` (1906 LOC)

---

_Architecture analysis: 2026-03-26_

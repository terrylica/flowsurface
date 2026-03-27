# Domain Pitfalls

**Domain:** Rust codebase refactoring (iced 0.14 GUI app, 3-crate workspace)
**Researched:** 2026-03-26

## Critical Pitfalls

Mistakes that cause compilation failures, runtime panics, or require reverting entire phases.

### Pitfall 1: RefCell Borrow Panics After Extracting Methods from canvas::Program

**What goes wrong:** The `canvas::Program` trait requires `&self` for `update()` and `draw()`. The codebase uses `RefCell<BarSelectionState>` inside `KlineChart` for interior mutability. When refactoring extracts helper functions from the monolithic `update()` or `draw()` methods, it is easy to accidentally hold a `RefCell::borrow()` across a call to a function that internally calls `borrow_mut()`. This compiles fine but panics at runtime.

**Why it happens:** In a single large method, the developer can see all borrow/borrow_mut calls and manually sequence them. Once logic is split across files (e.g., `bar_selection.rs`, `crosshair.rs`), the call chain is no longer visible in one screen. A function in `rendering.rs` may read `bar_selection` state (immutable borrow) while a function in `bar_selection.rs` needs to update it (mutable borrow). If the caller holds the immutable borrow when calling the mutable path, runtime panic.

**Consequences:** `BorrowMutError` panic kills the app. No compile-time warning. Manifests only when the specific user interaction path is exercised (e.g., Shift+Click during rendering), making it hard to catch without manual testing.

**Prevention:**

- Establish a strict rule: **extract the value, drop the borrow, then call borrow_mut()**. Document this in a code comment at every `RefCell` usage site.
- Before any method extraction from `canvas::Program` impls, grep for all `borrow()` and `borrow_mut()` calls on the same `RefCell` and map the call graph.
- Consider replacing `RefCell<BarSelectionState>` with `Cell<BarSelectionState>` if the state is `Copy`, or split it into multiple `Cell<T>` fields for individual primitives. `Cell` cannot panic.
- Add a smoke test: exercise every `canvas::Program::update()` interaction path (click, shift+click, drag, scroll) after each refactoring phase.

**Detection:** Runtime `BorrowMutError` panic. In CI, this is invisible without integration tests that exercise canvas interactions.

**Phase relevance:** Any phase that splits `kline/mod.rs` or `heatmap.rs`. Highest risk in the "split kline/mod.rs into canvas rendering + data operations" work.

---

### Pitfall 2: Visibility Downgrade Breaks Cross-Module Access After Splits

**What goes wrong:** Moving types or functions from a god module (e.g., `pane.rs`) into submodules and marking them `pub(super)` or `pub(crate)` can silently make them unreachable from other crates or distant modules, even though the compiler shows no error at the declaration site.

**Why it happens:** Rust visibility is path-dependent: a `pub(crate)` item in a private submodule is effectively private. When you split `pane.rs` into `pane/event_dispatch.rs` + `pane/chart_factory.rs`, the new submodules must be `pub(crate) mod` or have their items re-exported via `pub(crate) use`. If the parent `mod pane` is public but the submodule is private, `pub(crate)` items inside it are unreachable from outside `pane/`. [Kobzol's analysis](https://kobzol.github.io/rust/2025/04/23/two-ways-of-interpreting-visibility-in-rust.html) confirms this as a fundamental design gap.

**Consequences:** Compilation errors in distant files. Shotgun surgery to fix visibility. Temptation to "just make everything pub" which defeats the refactoring purpose.

**Prevention:**

- Use barrel re-exports at every split point. When splitting `pane.rs` into submodules, the parent `pane/mod.rs` must re-export all items that were previously accessible: `pub(crate) use event_dispatch::{...};`
- Enable `#[warn(unreachable_pub)]` and clippy's `redundant_pub_crate` lint during refactoring. These catch the exact case where `pub(crate)` is unreachable.
- Before splitting, run `cargo clippy` and record the baseline. After splitting, diff the warnings.
- Follow the existing convention in `src/chart.rs` which already demonstrates correct re-export patterns.

**Detection:** `cargo check` fails with "cannot find ... in module" errors in files that previously compiled.

**Phase relevance:** Every phase that creates new submodules, especially the pane.rs and kline/mod.rs splits.

---

### Pitfall 3: LazyLock Statics Silently Swallow Init Failures After Module Moves

**What goes wrong:** The codebase has ~20 `LazyLock` and `OnceLock` statics across `exchange/` (env var reads, HTTP clients, rate limiters). Moving these to a centralized config module can introduce initialization ordering issues or change when `std::env::var()` is called relative to `.mise.toml` environment setup.

**Why it happens:** `LazyLock` initializes on first access, not at program start. If statics are moved to a config module that is accessed earlier in the startup sequence (e.g., imported by `main.rs` before `mise` environment is loaded), `std::env::var()` returns `Err` and the `unwrap_or_else` fallback silently provides wrong defaults. The CONCERNS.md already flags "3 OnceLock statics with silent fallback on init failure" as temporal coupling.

**Consequences:** App starts with wrong ClickHouse host/port, wrong Telegram credentials, or wrong SSE config. No error message. Debugging requires checking every env var read site. If `LazyLock::new` panics, the lock is **permanently poisoned** -- all future accesses panic with no recovery path.

**Prevention:**

- When centralizing env vars into a config struct, use eager initialization at a known point in `main()` rather than lazy statics. Pattern: `Config::init()` called explicitly after runtime setup.
- If keeping `LazyLock`, add a `config::validate()` function called early in `main()` that forces all statics to initialize and logs their values. This already partially exists as `mise run preflight`.
- Never use `unwrap_or_else` with env vars in `LazyLock` -- use `unwrap_or("default")` with the default logged at INFO level so misconfiguration is visible.
- For the `ODB_SYMBOLS: OnceLock<Vec<String>>` pattern, ensure `init_odb_symbols()` is called before any code path that accesses it, or convert to `LazyLock` with explicit error propagation.

**Detection:** App connects to wrong ClickHouse, SSE, or Telegram endpoint. Manifests as "ClickHouse HTTP 404" or silent data absence. Hard to detect without startup validation.

**Phase relevance:** The "centralize all env var reads into a single config struct" phase. This is the single most likely phase to introduce subtle runtime bugs.

---

### Pitfall 4: Breaking the iced Task/Message Pipeline During Pane Event Dispatch Extraction

**What goes wrong:** The iced architecture routes all UI events through a `Message` enum -> `update()` -> `Task<Message>` pipeline. When extracting event dispatch logic from `pane.rs` (2425 LOC) into submodules, it is easy to break the return type chain: a helper function returns `Task::none()` instead of propagating the `Effect` or `Action` that the parent `update()` expected.

**Why it happens:** iced's `Task` type is opaque -- you cannot inspect what a task does. If an extracted function silently drops a `Task::perform(...)` and returns `Task::none()`, the effect (e.g., a ClickHouse fetch, a stream refresh) simply never happens. No compiler warning, no runtime error, just missing data. The existing `Action` and `Effect` enums in `pane.rs` are complex (4 `Action` variants, 4 `Effect` variants) and must be correctly plumbed through every extraction.

**Consequences:** Panes show "Fetching Klines..." forever, or streams fail to refresh after ticker switch, or chart state stops updating. Silent failures that appear only during specific user workflows.

**Prevention:**

- When extracting event dispatch, use `#[must_use]` on all `Action`, `Effect`, and `Task` return types. The compiler will warn if a return value is silently dropped.
- Write the extraction as a series of function renames first (move code, same signature, same call site) before changing any interfaces.
- After each extraction step, test the full pane lifecycle: open pane -> load data -> switch ticker -> scroll -> close pane.
- Keep a mapping document: "Message variant X -> which function handles it -> what Task/Effect it returns." Verify this mapping is preserved after extraction.

**Detection:** Functional regression -- pane stops loading data or responding to events. No compile error.

**Phase relevance:** The "split pane.rs into event dispatch + chart factory + interaction modules" phase.

---

## Moderate Pitfalls

### Pitfall 5: Borrow Checker Conflicts from Splitting Struct Access Across Modules

**What goes wrong:** When a god module like `kline/mod.rs` accesses multiple fields of `KlineChart` in a single method (e.g., `&self.chart` and `&mut self.bar_selection` simultaneously), extracting those field accesses into separate module functions forces them through `&self` / `&mut self` boundaries that the borrow checker cannot split.

**Why it happens:** Rust allows split borrows of struct fields within a single function: you can borrow `self.chart` immutably while borrowing `self.indicators` mutably. But the moment you pass `&mut self` to an extracted function, that function borrows the entire struct. This is the ["borrow is contagious"](https://qouteall.fun/qouteall-blog/2025/How%20to%20Avoid%20Fighting%20Rust%20Borrow%20Checker) problem.

**Prevention:**

- Pass individual fields as parameters rather than `&self`/`&mut self`. Example: `fn update_indicators(chart: &ViewState, indicators: &mut EnumMap<...>)` instead of `fn update_indicators(&mut self)`.
- Use the "destructure at call site" pattern: `let KlineChart { chart, indicators, .. } = self;` then pass the destructured fields to helpers.
- Avoid getter/setter methods on `KlineChart` that take `&self`/`&mut self` -- they prevent field-level split borrowing.

**Detection:** Compilation error: "cannot borrow `*self` as mutable because it is also borrowed as immutable."

**Phase relevance:** Any god module split, especially kline/mod.rs where `KlineChart` has 20+ fields.

---

### Pitfall 6: Re-export Breakage When Moving Types Between Crates

**What goes wrong:** Moving a type from one workspace crate to another (e.g., from `flowsurface` to `flowsurface-data`) and re-exporting it via `pub use` mostly works, but [breaks for unit structs and tuple structs](https://predr.ag/blog/moving-and-reexporting-rust-type-can-be-major-breaking-change/). `pub type Foo = inner::Foo;` (type alias) cannot be used as a constructor.

**Why it happens:** Rust treats type aliases differently from re-exports. `pub use` preserves constructor capability. `pub type` does not. Developers unfamiliar with this distinction may use type aliases when re-exporting moved types.

**Prevention:**

- Always use `pub use` for re-exports, never `pub type` when the type needs to be constructable.
- This project explicitly scopes out crate-level splits (PROJECT.md: "Not splitting workspace into more crates"), so this risk is lower. But within-crate module moves still require `pub use` re-exports at the old path.
- Verify after every move: `cargo check` + search for the moved type name to ensure all import sites resolve.

**Detection:** Compilation error at construction sites: "can't use a type alias as a constructor."

**Phase relevance:** Low risk given crate splits are out of scope. Relevant if types move between `mod` boundaries within a crate.

---

### Pitfall 7: ODB Triple-Stream Invariant Violated During Pane Refactoring

**What goes wrong:** ODB panes require exactly 3 streams (`OdbKline`, `Trades`, `Depth`) registered in `resolve_content()`. When refactoring pane.rs, if the stream registration logic is split into a separate module and a stream is accidentally omitted, the pane silently waits forever with no error.

**Why it happens:** There is no compile-time validation that ODB panes have all 3 streams. The requirement is documented in CLAUDE.md but enforced only by manual code review. Refactoring creates an opportunity to accidentally drop a stream registration line.

**Prevention:**

- Before refactoring pane.rs, add a debug assertion: `debug_assert!(streams.len() >= 3, "ODB panes require OdbKline + Trades + Depth")` in the ODB branch of `resolve_content()`.
- Create a `fn odb_streams(ticker: &TickerInfo, threshold: u32) -> Vec<StreamKind>` factory function that always returns all 3. Call this single function from wherever ODB panes are set up.
- After the pane split, verify by running the app and opening an ODB pane.

**Detection:** "Waiting for trades..." message that never resolves. Already documented as a known error pattern.

**Phase relevance:** The pane.rs split phase.

---

### Pitfall 8: Indicator Ceremony Reduction Creates New God Module

**What goes wrong:** The goal of reducing indicator addition from 36-file-touch to fewer touch points may lead to creating a new "indicator registry" god module that becomes the next maintenance burden. Over-abstracting the indicator factory pattern (traits, macros, auto-registration) adds indirection that makes debugging harder.

**Why it happens:** Natural tendency to centralize too aggressively. The current 36-file ceremony is painful but explicit. A macro-based or trait-object registry trades explicitness for magic.

**Prevention:**

- Target 5-8 touch points, not 1. Each should be obvious and documented with a checklist comment.
- Prefer a simple match-arm pattern over trait-object registries: match arms are compiler-checked, trait objects are not.
- Keep the `FOR_SPOT` and `FOR_PERPS` arrays as explicit lists rather than auto-discovery.
- Measure: after refactoring, adding a new indicator should require modifying N files, each with a clear `// ADD NEW INDICATOR HERE` comment.

**Detection:** Code review reveals new abstractions that are harder to understand than the old ceremony.

**Phase relevance:** The "reduce indicator addition ceremony" phase.

---

## Minor Pitfalls

### Pitfall 9: Stale Geometry Caches After Extracting Rendering Logic

**What goes wrong:** The 4-layer canvas cache (main, watermark, legend, crosshair) must be invalidated at the right times. When extraction moves rendering into submodules, the `clear()` calls on cache layers may be missed or called at wrong granularity (e.g., clearing `main` cache on every cursor move instead of just `crosshair`).

**Prevention:** Document cache invalidation rules as constants or comments at the cache declaration site. After extraction, grep for all `.clear()` calls and verify they match the invalidation table in CLAUDE.md.

**Detection:** Visual glitches: stale candles, crosshair lag, watermark disappearing. Performance regression if main cache clears too often.

**Phase relevance:** kline/mod.rs rendering extraction.

---

### Pitfall 10: Clippy -D warnings Fails on Unused Imports After Moves

**What goes wrong:** Moving functions between modules leaves behind unused `use` statements. With `clippy -D warnings`, this is a hard compile error, not a warning.

**Prevention:** Run `cargo clippy --fix --allow-dirty` after each move. Keep `mise run lint` as the verification gate after every atomic change. Do not batch multiple moves before checking -- each move should compile clean.

**Detection:** Immediate `cargo clippy` failure.

**Phase relevance:** Every phase. Low severity but high frequency.

---

### Pitfall 11: Async Task Lifetime Issues When Extracting Fetcher Logic

**What goes wrong:** `Task::sip()` and `Task::perform()` in `connector/fetcher.rs` capture references and closures. Moving these into separate modules can trigger lifetime issues if the closures reference data that doesn't live long enough in the new module context.

**Prevention:** When extracting async task construction, ensure all captured data is either `'static` or cloned into the closure. Use `move` closures explicitly. The existing pattern of `let ticker = ticker.clone()` before async blocks is correct -- preserve it during moves.

**Detection:** Compilation error about lifetimes in closures.

**Phase relevance:** Any phase touching `connector/fetcher.rs` or `main.rs` task dispatch.

---

## Phase-Specific Warnings

| Phase Topic                      | Likely Pitfall                                                | Mitigation                                               |
| -------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| Config centralization (env vars) | LazyLock init timing changes (#3)                             | Eager init in main(), validate all config at startup     |
| pane.rs split                    | Task/Effect chain breakage (#4), ODB stream invariant (#7)    | `#[must_use]` on returns, debug_assert on stream count   |
| kline/mod.rs split               | RefCell borrow panic (#1), borrow checker split failures (#5) | Extract values before borrow_mut, pass fields not self   |
| Indicator ceremony               | Over-abstraction (#8)                                         | Target 5-8 touch points, keep explicit match arms        |
| Exchange adapter dedup           | LazyLock statics shared across adapters                       | Verify each adapter's limiter static remains independent |
| Settings decoupling              | Visibility breakage (#2)                                      | Barrel re-exports, enable unreachable_pub lint           |

## Sources

- [Kobzol: Two ways of interpreting visibility in Rust](https://kobzol.github.io/rust/2025/04/23/two-ways-of-interpreting-visibility-in-rust.html) -- visibility pitfalls with `pub(crate)` in nested modules
- [Predrag: Moving and re-exporting a Rust type can be a major breaking change](https://predr.ag/blog/moving-and-reexporting-rust-type-can-be-major-breaking-change/) -- `pub type` vs `pub use` for constructors
- [Bevy Cheat Book: Split Borrows](https://bevy-cheatbook.github.io/pitfalls/split-borrows.html) -- borrow checker field splitting patterns
- [How to Avoid Fighting Rust Borrow Checker](https://qouteall.fun/qouteall-blog/2025/How%20to%20Avoid%20Fighting%20Rust%20Borrow%20Checker) -- contagious borrows, struct destructuring
- [Splitting Borrows - Rustonomicon](https://doc.rust-lang.org/nomicon/borrow-splitting.html) -- official guidance on field-level borrows
- [LazyLock documentation](https://doc.rust-lang.org/std/sync/struct.LazyLock.html) -- poisoning behavior, initialization semantics
- [RefCell - The Rust Book](https://doc.rust-lang.org/book/ch15-05-interior-mutability.html) -- runtime borrow checking rules

---

_Pitfalls audit: 2026-03-26_

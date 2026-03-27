# Technology Stack: Rust Refactoring Tooling

**Project:** flowsurface maintainability refactoring
**Researched:** 2026-03-26
**Focus:** Tools and patterns for refactoring a ~118-file Rust + iced 0.14 codebase

## Recommended Stack

### Structural Search & Transform

| Technology                                    | Purpose                                                       | Why                                                                                                                                                                                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ast-grep](https://crates.io/crates/ast-grep) | Structural code search and automated rewriting                | Pattern-based AST matching using tree-sitter. Write `std::env::var($KEY)` to find all scattered env reads. Write YAML rules to batch-rewrite import paths after module moves. Faster than manual find-replace, safer than regex because it understands syntax. |
| rust-analyzer (latest stable)                 | Interactive extract-function, extract-module, inline-variable | IDE-integrated refactoring for one-off extractions. "Extract function" handles borrow-checker-aware signature generation (mutable refs, lifetimes). Use for splitting god modules interactively, not for batch transforms.                                     |

**Confidence:** HIGH for ast-grep (verified via official site, active releases through March 2026). HIGH for rust-analyzer (standard Rust tooling).

### Code Analysis & Metrics

| Technology                                              | Purpose                                           | Why                                                                                                                                                                                                                                                              |
| ------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [cargo-modules](https://crates.io/crates/cargo-modules) | Module structure visualization + orphan detection | Three commands: `structure` (tree view), `dependencies` (internal dep graph), `orphans` (unlinked .rs files). Use before and after each split to verify module boundaries are cleaner. Critical for validating that god-module splits actually reduced coupling. |
| cargo clippy (bundled with rustc)                       | Lint gate: zero warnings policy                   | Already in use. Keep `cargo clippy -- -D warnings` as the compile-time quality gate after every refactoring step. No additional configuration needed.                                                                                                            |
| [cargo-machete](https://crates.io/crates/cargo-machete) | Detect unused dependencies                        | Fast ripgrep-based scan. Run after refactoring to catch dependencies that became unused when code moved between modules. Imprecise (misses macro-only usage) so verify flagged deps before removing.                                                             |

**Confidence:** HIGH for cargo-modules (verified GitHub, active maintenance). HIGH for clippy (standard). MEDIUM for cargo-machete (verified crates.io, known false-positive limitation with macro-based deps like `log`).

### API Compatibility Verification

| Technology                                                          | Purpose                                        | Why                                                                                                                                                                                                            |
| ------------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [cargo-semver-checks](https://crates.io/crates/cargo-semver-checks) | Verify pub API hasn't broken after refactoring | 245+ lints for semver violations. Relevant because `exchange/` and `data/` are workspace crates consumed by the main crate. After moving pub items between modules, run to confirm no accidental API breakage. |

**Confidence:** MEDIUM. This project doesn't publish to crates.io and the workspace crates are internal, so semver-checks provides a safety net but isn't critical. The real guard is `cargo clippy -- -D warnings` catching unresolved imports.

**Recommendation:** Install it, run it once after each module split phase, but don't make it a blocking gate. If it flags something, that means a re-export is missing.

### Dependency Visualization

| Technology                                                | Purpose                   | Why                                                                                                                                                               |
| --------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cargo tree (bundled)                                      | External dependency tree  | Built-in. Use `cargo tree -d` to find duplicate dependency versions. Use `cargo tree -i <crate>` to understand why a dep exists.                                  |
| [cargo-depgraph](https://crates.io/crates/cargo-depgraph) | Graphviz dependency graph | Visual workspace crate graph. Less useful here (only 3 workspace crates) but helpful to verify that module splits haven't introduced unexpected cross-crate deps. |

**Confidence:** HIGH for cargo tree (built-in). LOW for cargo-depgraph (nice-to-have, not essential for a 3-crate workspace).

## What NOT to Use

| Tool                               | Why Not                                                                                                                                           |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| cargo-udeps                        | Slower than cargo-machete (requires full compilation). cargo-machete's speed matters more for a refactoring workflow where you run it frequently. |
| cargo-geiger                       | Detects unsafe code -- irrelevant to this refactoring. The codebase uses iced/wgpu which have their own unsafe; auditing that is out of scope.    |
| cargo-bloat                        | Binary size analysis -- irrelevant to structural refactoring.                                                                                     |
| Crate-level splitting tools        | PROJECT.md explicitly rules out crate splits. Module-level is the target.                                                                         |
| DI frameworks (shaku, etc.)        | Rust's `LazyLock` + module-level organization is idiomatic. DI frameworks add indirection without benefit for this codebase size.                 |
| Procedural macro crates for config | Overkill. A simple `Config` struct with `LazyLock` initialization covers the scattered env-var problem.                                           |

## Installation

```bash
# Structural search (install via cargo or brew)
cargo install ast-grep --locked
# OR: brew install ast-grep

# Module analysis
cargo install cargo-modules --locked

# Unused dependency detection
cargo install cargo-machete --locked

# API compatibility (optional safety net)
cargo install cargo-semver-checks --locked

# Dependency graph (optional)
cargo install cargo-depgraph --locked
```

## Tool Workflow Per Refactoring Phase

### Before Each Module Split

```bash
# 1. Visualize current structure
cargo modules structure --package flowsurface --lib 2>/dev/null || \
  cargo modules structure --package flowsurface

# 2. Find all references to the pattern you're extracting
ast-grep --pattern 'std::env::var($KEY)' --lang rust

# 3. Baseline: confirm clean compile
cargo clippy -- -D warnings
```

### During Module Extraction

```bash
# Use rust-analyzer "Extract module" in IDE for interactive splits
# Use ast-grep for batch import path rewrites after moves:
ast-grep --pattern 'use crate::screen::dashboard::pane::$ITEM' \
  --rewrite 'use crate::screen::dashboard::pane::event::$ITEM' \
  --lang rust
```

### After Each Module Split

```bash
# 1. Verify no orphan files
cargo modules orphans --package flowsurface

# 2. Verify clean compile
cargo clippy -- -D warnings

# 3. Check for newly-unused deps
cargo machete

# 4. Optional: verify pub API intact across workspace crates
cargo semver-checks check-release --package exchange
cargo semver-checks check-release --package data
```

## ast-grep Patterns for This Project

### Finding Scattered env::var Reads

```yaml
# .ast-grep/find-env-reads.yaml
id: find-env-reads
language: Rust
rule:
  pattern: std::env::var($KEY)
```

```bash
ast-grep --pattern 'std::env::var($KEY)' --lang rust
```

### Finding God Module Imports (Feature Envy Detection)

```bash
# Find files importing many items from a single module
ast-grep --pattern 'use crate::screen::dashboard::pane::$$$ITEMS' --lang rust
```

### Rewriting Import Paths After Module Move

```yaml
# After moving event handling from pane.rs to pane/event.rs
id: rewrite-pane-event-imports
language: Rust
rule:
  pattern: use crate::screen::dashboard::pane::$ITEM
  inside:
    kind: use_declaration
fix: use crate::screen::dashboard::pane::event::$ITEM
```

**Note:** ast-grep rewrites are powerful but not infallible with Rust's complex module system. Always compile after batch rewrites. Use `--interactive` mode for review before applying.

## iced 0.14 Architectural Constraints on Refactoring

### What You Can Split

| Component             | Splittable?                                                               | Constraint                                                                                                           |
| --------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `update()` logic      | YES -- extract message handlers into sub-functions                        | Each handler is a pure `&mut self` + `Message` -> `Task<Message>` function. Can live in separate modules.            |
| `view()` logic        | YES -- extract widget builders into separate modules                      | View functions are pure `&self` -> `Element<Message>`. Easily extractable.                                           |
| Canvas `Program` impl | PARTIALLY -- extract draw helpers, keep `Program` trait impl in one place | `draw()` and `update()` must be on the same `impl Program for T`. Helper functions can be in submodules.             |
| `subscription()`      | YES -- extract into per-stream subscription builders                      | Each stream subscription is independent.                                                                             |
| Message enum          | CAREFULLY -- nested enums are idiomatic                                   | Split into `PaneMessage`, `ChartMessage`, etc. But the top-level `Message` must be in scope for `iced::Application`. |

### What You Cannot Split

| Component                   | Why Not                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `Application` trait impl    | Must be a single `impl Application for App`. The `update`, `view`, `subscription`, `theme` methods must all be on one type.          |
| Canvas `Program` trait impl | `update()` takes `&self` (not `&mut self`). Interior mutability via `RefCell` is required. Cannot split the trait impl across files. |
| `Task<Message>` return type | All update paths must return the same `Task<Message>` type. Message routing must stay unified.                                       |

### Idiomatic iced Module Pattern

```
src/
  app.rs              # Application trait impl (thin dispatcher)
  message.rs          # Top-level Message enum + nested sub-enums
  screen/
    dashboard.rs      # Dashboard state + update + view (dispatcher)
    dashboard/
      pane/
        mod.rs        # PaneState, minimal dispatch
        event.rs      # Event handling (extracted from god module)
        factory.rs    # Pane creation / chart factory
        interaction.rs # Mouse/keyboard interaction
  chart/
    kline/
      mod.rs          # KlineChart state, Program trait impl (kept unified)
      render.rs       # Draw helpers (extracted from god module)
      data_ops.rs     # Data manipulation (trade insertion, aggregation)
      legend.rs       # Legend/watermark drawing
```

The key insight: iced's `Program` trait forces the trait impl to stay in one file, but all the heavy logic it calls can be in submodules. The `mod.rs` becomes a thin dispatcher that delegates to extracted modules.

## Sources

- [ast-grep official site](https://ast-grep.github.io/) -- tool docs, Rust catalog
- [ast-grep GitHub](https://github.com/ast-grep/ast-grep) -- latest releases
- [ast-grep Rust catalog](https://ast-grep.github.io/catalog/rust/) -- Rust-specific rules
- [cargo-semver-checks GitHub](https://github.com/obi1kenobi/cargo-semver-checks)
- [cargo-semver-checks 2025 year in review](https://predr.ag/blog/cargo-semver-checks-2025-year-in-review/) -- 245 lints
- [cargo-modules GitHub](https://github.com/regexident/cargo-modules) -- structure/deps/orphans
- [cargo-machete GitHub](https://github.com/bnjbvr/cargo-machete) -- unused dep detection
- [iced architecture book](https://book.iced.rs/architecture.html) -- Elm Architecture constraints
- [iced 0.14 HN discussion](https://news.ycombinator.com/item?id=46185323) -- community patterns
- [rust-analyzer extract function](https://github.com/rust-lang/rust-analyzer/blob/master/crates/ide-assists/src/handlers/extract_function.rs) -- IDE refactoring

---

_Stack analysis: 2026-03-26_

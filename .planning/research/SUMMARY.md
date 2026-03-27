# Research Summary: Flowsurface Refactoring Tooling

**Domain:** Rust codebase structural refactoring (iced 0.14 GUI desktop app)
**Researched:** 2026-03-26
**Overall confidence:** HIGH

## Executive Summary

The flowsurface codebase has clear structural problems identified by a 4-agent audit: two god modules (pane.rs at 2425 LOC, kline/mod.rs at 2388 LOC), scattered env var reads across 6 files, and a 36-file ceremony for adding indicators. The refactoring tooling ecosystem for Rust in 2025-2026 is mature enough to support this work systematically.

The recommended tooling stack is intentionally minimal: ast-grep for structural search and batch import rewrites, rust-analyzer for interactive extract-function/module operations, cargo-modules for before/after module structure validation, and cargo clippy as the compile-time quality gate. cargo-semver-checks and cargo-machete serve as optional safety nets. No new frameworks, no DI, no proc macros -- the refactoring is purely structural module splitting using idiomatic Rust patterns.

The most important architectural constraint comes from iced 0.14 itself: the `canvas::Program` trait requires `&self` (not `&mut self`), forcing interior mutability via `RefCell`. This means the `Program` trait impl must stay in a single file to maintain visibility over borrow/borrow_mut discipline. All other logic (data operations, ODB lifecycle, view helpers) can be freely extracted into submodules using Rust's impl-block-per-file pattern.

The refactoring should proceed config-first (lowest risk, highest immediate value), then god-module splits (pane.rs before kline/mod.rs due to higher daily pain), then indicator ceremony reduction. Exchange adapter deduplication is deferred -- the adapters work and change rarely.

## Key Findings

**Stack:** ast-grep + rust-analyzer + cargo-modules + cargo clippy. No new dependencies. Pure structural tooling.
**Architecture:** Split by feature domain (iced maintainer recommendation), not by layer. Impl-block-per-file pattern for struct method distribution.
**Critical pitfall:** RefCell borrow panics after extracting methods from canvas::Program -- runtime-only failure, no compile-time warning.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Config centralization** - Lowest risk, highest immediate clarity
   - Addresses: Scattered env reads (6 files, 4 duplicates), OnceLock temporal coupling
   - Avoids: LazyLock init timing pitfall (use eager init in main())

2. **God module split: pane.rs** - Highest daily pain point
   - Addresses: 2425 LOC event dispatch + chart factory + UI rendering
   - Avoids: Task/Effect chain breakage, ODB triple-stream invariant

3. **God module split: kline/mod.rs** - Second highest LOC, rendering clarity
   - Addresses: 2388 LOC canvas rendering + data operations
   - Avoids: RefCell borrow panics (keep Program impl unified)

4. **Indicator ceremony reduction** - Highest leverage for future development
   - Addresses: 36-file touch for new indicators
   - Avoids: Over-abstraction (target 5-8 touch points, not 1)

5. **Settings decoupling + bool elimination** - Low-risk cleanup
   - Addresses: Feature envy, bool flag arguments
   - Avoids: Visibility breakage from nested module moves

**Phase ordering rationale:**

- Config first because every subsequent phase benefits from centralized config
- pane.rs before kline/mod.rs because pane is touched more frequently in daily development
- Indicator ceremony after god module splits because the splits may change the indicator registration paths
- Exchange adapter dedup deferred indefinitely (adapters are stable, duplication is tolerable)

**Research flags for phases:**

- Phase 3 (kline/mod.rs): Likely needs deeper research on RefCell discipline patterns
- Phase 4 (indicators): Needs concrete analysis of which 36 files are touched and which can be consolidated
- Phases 1, 2, 5: Standard patterns, unlikely to need additional research

## Confidence Assessment

| Area                        | Confidence | Notes                                                                                               |
| --------------------------- | ---------- | --------------------------------------------------------------------------------------------------- |
| Stack (tooling)             | HIGH       | ast-grep, cargo-modules, cargo-semver-checks all verified current via official sources              |
| Features (what to refactor) | HIGH       | Based on concrete codebase audit with LOC counts and file evidence                                  |
| Architecture (how to split) | HIGH       | iced maintainer guidance on feature-based splitting verified; impl-block-per-file is idiomatic Rust |
| Pitfalls (what to avoid)    | HIGH       | RefCell, visibility, LazyLock timing all documented with Rust-specific sources                      |

## Gaps to Address

- **No integration test coverage**: The codebase has zero tests for ODB reconciliation, pane streams, or gap-fill. Refactoring without tests relies entirely on manual smoke testing + compiler checks.
- **Indicator ceremony specifics**: The "36 files" number needs decomposition into which files are truly necessary vs. which can be consolidated. Phase-specific research needed.
- **Dashboard split decision**: dashboard.rs (1906 LOC) is borderline. Research recommends deferring (upstream-controlled), but this may need revisiting if pane.rs split reveals dashboard as the new bottleneck.

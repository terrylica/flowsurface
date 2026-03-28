# Phase 8: Indicator Ceremony Reduction - Research

**Researched:** 2026-03-27
**Domain:** Rust enum-dispatch indicator system, iced canvas architecture
**Confidence:** HIGH

## Summary

The "36 file" indicator ceremony claim from the 4-agent audit is FALSE for the current codebase. The RSI commit (0d21c59) touched 36 files, but only **3 files** contained actual indicator registration changes. The remaining 33 files were a bulk `rangebar-py` -> `opendeviationbar-py` comment rename that was bundled into the same commit.

After Phases 1-7 restructuring, the actual indicator ceremony for a standard subplot indicator is **3 files** (2 modified + 1 new). This already exceeds the "6 or fewer" success criterion. The phase can be satisfied primarily through **documentation** (a verified checklist) plus minor consolidation of the two factory functions.

**Primary recommendation:** Document the actual 3-file ceremony as the official checklist. Merge the two factory functions (`make_empty` and `make_indicator_with_config`) into a single `make_indicator` function to eliminate a confusing duplication. The goal of 6-or-fewer files is already achieved.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- all implementation choices at Claude's discretion.

### Claude's Discretion

All implementation choices are at Claude's discretion -- pure infrastructure phase. Key constraints from ROADMAP success criteria:

1. A documented checklist exists showing exactly which files to touch for a new indicator (6 or fewer)
2. The indicator registration path uses explicit match arms (not trait objects or macros) -- compiler-checked, not magic
3. FOR_SPOT and FOR_PERPS arrays remain as explicit lists

### Deferred Ideas (OUT OF SCOPE)

None.
</user_constraints>

<phase_requirements>

## Phase Requirements

| ID      | Description                                                                 | Research Support                                                                                 |
| ------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| QUAL-03 | Indicator addition ceremony reduced from 36 file touch points to 6 or fewer | Current ceremony is already 3 files; documented checklist + factory consolidation completes this |
| VER-01  | `cargo clippy -- -D warnings` passes after every phase                      | Standard verification; enum_map derive handles exhaustiveness                                    |
| VER-02  | Zero behavior changes -- all existing functionality works identically       | Factory consolidation is pure refactor; no indicator behavior changes                            |
| VER-03  | No new `unsafe` code introduced                                             | No unsafe needed for this work                                                                   |

</phase_requirements>

## Standard Stack

No new libraries needed. This phase is pure code organization and documentation.

### Core (already in project)

| Library  | Version | Purpose                                                      | Why Standard                                                          |
| -------- | ------- | ------------------------------------------------------------ | --------------------------------------------------------------------- |
| enum-map | 2.7.3   | EnumMap<KlineIndicator, Option<Box<dyn KlineIndicatorImpl>>> | Zero-cost enum-keyed map; derive macro auto-updates when enum changes |

**Key insight:** The `enum_map::Enum` derive on `KlineIndicator` means the compiler already enforces that `EnumMap` storage adapts automatically when variants are added. No manual map registration needed.

## Architecture Patterns

### Current Indicator Registration Path (3 files)

For a **standard subplot indicator** (like Volume, Delta, TradeCount, RSI):

**File 1: `data/src/chart/indicator.rs`** (enum definition)

- Add variant to `KlineIndicator` enum
- Add to `FOR_SPOT` and/or `FOR_PERPS` arrays (update array length const)
- Add `Display` match arm
- If overlay (not subplot): add to `has_subplot()` exclusion list

**File 2: `src/chart/indicator/kline.rs`** (factory + module registry)

- Add `pub mod <name>;` declaration
- Add match arm to `make_empty()` factory

**File 3: `src/chart/indicator/kline/<name>.rs`** (NEW FILE -- implementation)

- Implement `KlineIndicatorImpl` trait
- At minimum: `clear_all_caches`, `clear_crosshair_caches`, `element`, `rebuild_from_source`

**That's it.** The `EnumMap` derive handles storage allocation. The `for_market()` method handles UI menu filtering. Serialization is handled by serde derive on the enum.

### Extended Ceremony (4-5 files, only for configurable/special indicators)

Additional files are needed ONLY when the indicator has:

**Configurable parameters** (e.g., OFI EMA period, intensity lookback):

- `data/src/chart/kline.rs` -- Add field to `Config` struct (File 4)
- `src/chart/kline/mod.rs` -- Add arm to `make_indicator_with_config()` and add `set_<param>()` method (File 5)

**Special rendering on main canvas** (e.g., TradeIntensityHeatmap colors candle bodies):

- `src/chart/kline/mod.rs` -- Add indicator-specific lookup in draw code (File 5)
- `src/screen/dashboard/pane/content.rs` -- Special-case subplot counting (File 6)

### The Two Factory Functions (consolidation target)

Currently there are TWO factory functions that construct indicators:

1. `make_empty(which)` in `src/chart/indicator/kline.rs` -- exhaustive match, default construction
2. `make_indicator_with_config(which, cfg)` in `src/chart/kline/mod.rs` -- partial match with fallthrough to `make_empty`

This is confusing. `make_indicator_with_config` handles OFI, OFICumulativeEma, and TradeIntensityHeatmap specially, then falls through to `make_empty` for everything else.

**Consolidation:** Merge into a single `make_indicator(which, cfg)` in `src/chart/indicator/kline.rs`. This:

- Eliminates the confusing two-function pattern
- Keeps the single exhaustive match (compiler-checked)
- Reduces ceremony for configurable indicators by 1 file
- Does not change behavior (same construction, same config passthrough)

### Recommended Project Structure (unchanged)

```
data/src/chart/indicator.rs       # Enum definition, FOR_SPOT, FOR_PERPS, Display
src/chart/indicator/kline.rs      # Factory function, pub mod declarations, trait def
src/chart/indicator/kline/*.rs    # Individual indicator implementations
```

### Anti-Patterns to Avoid

- **Proc macro registration:** Violates success criteria ("explicit match arms, not macros")
- **Trait object auto-discovery:** Violates success criteria ("compiler-checked, not magic")
- **Inventory/linkme crates:** Runtime registration, not compile-time checked
- **Collapsing FOR_SPOT/FOR_PERPS into attributes:** Success criteria says "remain as explicit lists"

## Don't Hand-Roll

| Problem               | Don't Build                   | Use Instead                          | Why                                                   |
| --------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------------------- |
| Indicator storage map | HashMap<String, Box<dyn ...>> | EnumMap<KlineIndicator, Option<...>> | Already in place; zero-cost, compile-time checked     |
| Auto-registration     | Proc macro registry           | Explicit match arms                  | Success criteria mandate; compiler finds missing arms |

## Common Pitfalls

### Pitfall 1: Misidentifying the ceremony count

**What goes wrong:** Counting files from the RSI commit (36) without analyzing which changes were actually indicator-related
**Why it happens:** The RSI commit bundled a bulk comment rename with the actual feature
**How to avoid:** The research has already resolved this -- actual ceremony is 3 files
**Warning signs:** Any plan that claims "36 files need reduction"

### Pitfall 2: FOR_SPOT/FOR_PERPS array length constants

**What goes wrong:** Forgetting to update the array length (e.g., `[KlineIndicator; 9]` -> `[KlineIndicator; 10]`)
**Why it happens:** Rust const arrays require explicit length
**How to avoid:** The compiler catches this immediately -- not a silent bug
**Warning signs:** None needed; compile error is the safety net

### Pitfall 3: Factory function confusion

**What goes wrong:** Adding a configurable indicator to `make_empty` but not to `make_indicator_with_config`, resulting in config being silently ignored
**Why it happens:** Two factory functions with overlapping responsibility
**How to avoid:** Consolidate into single factory
**Warning signs:** Indicator works but ignores user's config settings

### Pitfall 4: EnumMap auto-expansion

**What goes wrong:** Assuming you need to manually register the new indicator in EnumMap storage
**Why it happens:** Not understanding that `#[derive(Enum)]` handles this
**How to avoid:** Just add the variant; EnumMap grows automatically
**Warning signs:** Someone adding manual map insertion code

## Code Examples

### Adding a Standard Subplot Indicator (current 3-file ceremony)

**File 1: `data/src/chart/indicator.rs`**

```rust
// 1. Add variant (inside the enum, before closing brace)
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, Eq, Enum)]
pub enum KlineIndicator {
    // ... existing variants ...
    /// New indicator description.
    NewIndicator,
}

// 2. Update FOR_SPOT array (increment length, add variant)
const FOR_SPOT: [KlineIndicator; 10] = [  // was 9
    // ... existing ...
    KlineIndicator::NewIndicator,
];

// 3. Update FOR_PERPS array (increment length, add variant)
const FOR_PERPS: [KlineIndicator; 11] = [  // was 10
    // ... existing ...
    KlineIndicator::NewIndicator,
];

// 4. Add Display arm
impl Display for KlineIndicator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // ... existing ...
            KlineIndicator::NewIndicator => write!(f, "New Indicator"),
        }
    }
}
```

**File 2: `src/chart/indicator/kline.rs`**

```rust
// 1. Add module declaration
pub mod new_indicator;

// 2. Add arm to make_empty (or consolidated make_indicator)
KlineIndicator::NewIndicator => {
    Box::new(super::kline::new_indicator::NewIndicatorImpl::new())
}
```

**File 3: `src/chart/indicator/kline/new_indicator.rs`** (new file)

```rust
// Implement KlineIndicatorImpl trait
// See rsi.rs (245 LOC) or volume.rs for reference implementations
```

### Factory Consolidation Pattern

```rust
// BEFORE: two functions
// src/chart/indicator/kline.rs
pub fn make_empty(which: KlineIndicator) -> Box<dyn KlineIndicatorImpl> { ... }

// src/chart/kline/mod.rs
fn make_indicator_with_config(which: KlineIndicator, cfg: &Config) -> Box<dyn KlineIndicatorImpl> {
    match which {
        KlineIndicator::OFI => /* special */ ,
        KlineIndicator::OFICumulativeEma => /* special */ ,
        KlineIndicator::TradeIntensityHeatmap => /* special */ ,
        other => indicator::kline::make_empty(other),  // fallthrough
    }
}

// AFTER: single function in src/chart/indicator/kline.rs
pub fn make_indicator(
    which: KlineIndicator,
    cfg: &data::chart::kline::Config,
) -> Box<dyn KlineIndicatorImpl> {
    match which {
        KlineIndicator::Volume => Box::new(volume::VolumeIndicator::new()),
        KlineIndicator::OpenInterest => Box::new(open_interest::OpenInterestIndicator::new()),
        KlineIndicator::Delta => Box::new(delta::DeltaIndicator::new()),
        KlineIndicator::TradeCount => Box::new(trade_count::TradeCountIndicator::new()),
        KlineIndicator::OFI => Box::new(ofi::OFIIndicator::with_ema_period(cfg.ofi_ema_period)),
        KlineIndicator::OFICumulativeEma => Box::new(
            ofi_cumulative_ema::OFICumulativeEmaIndicator::with_ema_period(cfg.ofi_ema_period),
        ),
        KlineIndicator::TradeIntensity => Box::new(trade_intensity::TradeIntensityIndicator::new()),
        KlineIndicator::TradeIntensityHeatmap => Box::new(
            trade_intensity_heatmap::TradeIntensityHeatmapIndicator::with_config(
                cfg.intensity_lookback, cfg.anomaly_fence,
            ),
        ),
        KlineIndicator::ZigZag => Box::new(zigzag::ZigZagOverlayIndicator::new()),
        KlineIndicator::RSI => Box::new(rsi::RsiIndicator::new()),
    }
}
```

## Validation Architecture

### Test Framework

| Property           | Value                                                                 |
| ------------------ | --------------------------------------------------------------------- |
| Framework          | cargo test (Rust built-in)                                            |
| Config file        | Cargo.toml workspace                                                  |
| Quick run command  | `cargo clippy --all-targets -- -D warnings`                           |
| Full suite command | `cargo clippy --all-targets -- -D warnings && cargo test --workspace` |

### Phase Requirements -> Test Map

| Req ID  | Behavior                               | Test Type | Automated Command                                          | File Exists?         |
| ------- | -------------------------------------- | --------- | ---------------------------------------------------------- | -------------------- |
| QUAL-03 | Indicator ceremony is 6 or fewer files | manual    | Documented checklist in CLAUDE.md                          | N/A                  |
| VER-01  | Clippy passes clean                    | lint      | `cargo clippy --all-targets -- -D warnings`                | N/A (cargo built-in) |
| VER-02  | Zero behavior change                   | smoke     | `cargo build` (compiles = same behavior for pure refactor) | N/A                  |
| VER-03  | No new unsafe                          | lint      | `cargo clippy --all-targets -- -D warnings`                | N/A                  |

### Sampling Rate

- **Per task commit:** `cargo clippy --all-targets -- -D warnings`
- **Per wave merge:** `cargo clippy --all-targets -- -D warnings && cargo test --workspace`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

None -- existing build infrastructure covers all phase requirements. This is a refactor + documentation phase; clippy exhaustive match checking is the primary verification.

## Open Questions

1. **Should the checklist live in CLAUDE.md or a standalone doc?**
   - What we know: CLAUDE.md already has an "Adding a New Indicator" section (6 steps listed)
   - What's unclear: Whether to update that section or create a separate `docs/indicators/ADDING.md`
   - Recommendation: Update the existing CLAUDE.md section -- it's the canonical developer reference. The 6-step list there is stale (references pre-Phase-1-7 structure).

2. **Should `make_empty` be removed entirely or kept as a convenience?**
   - What we know: `make_empty` is used in one place (the fallthrough in `make_indicator_with_config`). After consolidation it has no callers.
   - What's unclear: Whether any future code path needs no-config construction
   - Recommendation: Remove it. The consolidated `make_indicator` always takes config (which has defaults via `#[serde(default)]`). If a caller doesn't have config, pass `&Config::default()`.

## Sources

### Primary (HIGH confidence)

- Direct codebase analysis of `data/src/chart/indicator.rs`, `src/chart/indicator/kline.rs`
- Git diff analysis of commit 0d21c59 (RSI addition) -- verified 33/36 files were comment renames
- Post-Phase-7 file structure verification

### Secondary (MEDIUM confidence)

- CLAUDE.md "Adding a New Indicator" pattern (6 steps -- stale, needs update)

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - direct codebase analysis, no external dependencies
- Architecture: HIGH - traced every `KlineIndicator::` reference across codebase
- Pitfalls: HIGH - verified through RSI commit forensics

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable -- this is internal architecture, not external dependency)

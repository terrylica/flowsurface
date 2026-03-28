# Phase 3: Bool-to-Enum Cleanup - Research

**Researched:** 2026-03-27
**Domain:** Rust API design -- replacing bool flag parameters with self-documenting enums
**Confidence:** HIGH

## Summary

Phase 3 targets 5 specific bool flag parameters across the codebase where call sites read as bare `true`/`false` rather than self-documenting enum variants. The scope is tightly bounded: define new enums, update function signatures, and fix all call sites. No behavioral changes.

The codebase already demonstrates this pattern well -- `data/src/panel/ladder.rs` has a `Side` enum (Bid/Ask) that wraps an `is_bid()` helper, and `exchange/src/unit/qty.rs` has `SizeUnit` (Base/Quote). The refactoring extends this established convention to 5 remaining bool flags.

**Primary recommendation:** Define each enum adjacent to the function that uses it (same module), update signatures, then mechanically fix call sites. One plan is sufficient -- all 5 targets are independent and low-risk.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

None -- all implementation choices are at Claude's discretion.

### Claude's Discretion

All implementation choices are at Claude's discretion -- pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions. Enum names should be descriptive and follow Rust naming conventions (PascalCase variants).

### Deferred Ideas (OUT OF SCOPE)

None -- infrastructure phase.
</user_constraints>

<phase_requirements>

## Phase Requirements

| ID      | Description                                                                                                                                     | Research Support                                                                             |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| QUAL-01 | 5 bool flag arguments replaced with enums or split into separate functions (adapter.rs, conditional_ema.rs, heatmap.rs, odb_core.rs, ladder.rs) | All 5 targets identified with exact signatures, call sites enumerated, enum designs proposed |
| VER-01  | `cargo clippy -- -D warnings` passes after every phase                                                                                          | Mechanical refactoring -- clippy compliance verified by updating all call sites              |
| VER-02  | Zero behavior changes -- all existing functionality works identically                                                                           | Pure type-level change: enum variant maps 1:1 to previous bool value                         |
| VER-03  | No new `unsafe` code introduced                                                                                                                 | No unsafe needed for enum definitions                                                        |

</phase_requirements>

## Target Inventory

### Target 1: `MarketKind::qty_in_quote_value()` -- adapter.rs

**File:** `exchange/src/adapter.rs:237`
**Current signature:** `pub fn qty_in_quote_value<T>(&self, qty: T, price: Price, size_in_quote_ccy: bool) -> f32`
**Bool meaning:** `true` = quantity already in quote currency, `false` = quantity in base currency (multiply by price)

**Existing enum:** `SizeUnit` already exists at `exchange/src/unit/qty.rs:16` with `Base` and `Quote` variants. Every call site already computes `let size_in_quote_ccy = volume_size_unit() == SizeUnit::Quote;` before passing the bool.

**Recommendation:** Change parameter from `bool` to `SizeUnit`. Call sites simplify from `volume_size_unit() == SizeUnit::Quote` bool intermediate to passing `volume_size_unit()` directly.

**Call sites (7):**

- `src/screen/dashboard/panel/timeandsales.rs` (3 sites)
- `src/chart/heatmap.rs` (2 sites)
- `data/src/chart/heatmap.rs` (2 sites)

**Ripple:** ~20 exchange adapter files also compute `size_in_quote_ccy` the same way for `QtyNormalizer::new()`. Those are separate functions (`QtyNormalizer`) and not in scope for this target, but the intermediate variable disappears at the 7 `qty_in_quote_value` call sites.

### Target 2: `ConditionalEma::update()` -- conditional_ema.rs

**File:** `data/src/conditional_ema.rs:22`
**Current signature:** `pub fn update(&mut self, input: f32, active: bool) -> f32`
**Bool meaning:** `true` = update EMA with new value, `false` = carry forward unchanged

**Analysis:** The sole call site (`src/chart/indicator/kline/ofi_cumulative_ema.rs:86`) always passes `true`. The `active` parameter was designed for future use (separate bullish/bearish EMAs) but is currently dead code for the `false` branch.

**Recommendation:** Define `enum EmaAction { Update, CarryForward }` in `conditional_ema.rs`. Even though only `Update` is currently used, the enum makes the API intent clear for future callers. The sole call site changes from `self.ema.update(ofi, true)` to `self.ema.update(ofi, EmaAction::Update)`.

**Call sites (1):**

- `src/chart/indicator/kline/ofi_cumulative_ema.rs:86`

### Target 3: `OrderRun` / `HistoricalDepth` -- heatmap.rs (data crate)

**File:** `data/src/chart/heatmap.rs`
**Functions affected:**

- `OrderRun::new(start_time, aggr_time, qty, is_bid: bool)` (line 122)
- `OrderRun.is_bid: bool` (struct field, line 118)
- `HistoricalDepth::process_side(&mut self, side, time, is_bid: bool)` (line 174)
- `HistoricalDepth::update_price_level(&mut self, time, price, qty, is_bid: bool)` (line 198)
- `CoalescingRun.is_bid: bool` (struct field, line 535)

**Bool meaning:** `true` = bid side, `false` = ask side

**Existing precedent:** `data/src/panel/ladder.rs:43` already has `pub enum Side { Bid, Ask }` with `is_bid()` helper. However, `Side` is currently scoped to the ladder module.

**Recommendation:** Promote `Side` enum to a shared location (e.g., `data/src/lib.rs` or a new `data/src/types.rs`) since both heatmap and ladder need it. Alternatively, since the `is_bid` field is on `OrderRun` (a serialized struct), the field needs to remain `is_bid: bool` for serde compatibility with existing saved state, OR use `#[serde(rename)]`. The cleaner path: change struct fields to `side: Side` with serde attributes, and update the function parameters.

**Important caveat:** `OrderRun` and `CoalescingRun` derive `Serialize`/`Deserialize`. The `is_bid: bool` field is serialized. Changing to `side: Side` requires either:

1. `#[serde(alias = "is_bid")]` for backward compat + new field name, OR
2. Custom serde logic

The `HistoricalDepth` data is ephemeral (not persisted to saved-state.json -- it's runtime depth data), so no migration needed. But `CoalescingRun` appears in `CoalesceKind` which may be serialized. Need to verify.

**Call sites:**

- `HistoricalDepth::insert_latest_depth()` at line 170-171: passes `true`/`false` directly
- `process_side` / `update_price_level`: internal propagation
- `depth_color()` in `src/chart/heatmap.rs:916`: uses `is_bid` from struct field
- Multiple reads of `run.is_bid` / `visual_run.is_bid` in rendering code

### Target 4: `insert_raw_trades()` -- odb_core.rs

**File:** `src/chart/kline/odb_core.rs:1259`
**Current signature:** `pub fn insert_raw_trades(&mut self, raw_trades: Vec<Trade>, is_batches_done: bool)`
**Bool meaning:** `true` = this is the final batch of gap-fill trades (finalize), `false` = more batches coming

**Recommendation:** Define `enum GapFillProgress { Streaming, Complete }` in `odb_core.rs`. Call site changes from `insert_raw_trades(trades, is_batches_done)` to `insert_raw_trades(trades, GapFillProgress::Complete)`.

**Call sites (1):**

- `src/screen/dashboard.rs:1124` (passes variable `is_batches_done`)
- The variable comes from `Message::TradesReceived { is_batches_done, .. }` at line 1102

**Ripple:** The `Message::TradesReceived` variant in `dashboard.rs` also carries this bool. The enum should propagate to the message variant field too.

### Target 5: `ChaseTracker::update()` -- ladder.rs

**File:** `data/src/panel/ladder.rs:218`
**Current signature:** `pub fn update(&mut self, current_best: Option<Price>, is_bid: bool, now_ms: u64, max_interval: Duration)`
**Bool meaning:** `true` = tracking bid side (up = chase), `false` = tracking ask side (down = chase)

**Recommendation:** Reuse the `Side` enum (same module already defines it). Change `is_bid: bool` to `side: Side`. Internal logic changes `if is_bid { Direction::Up } else { Direction::Down }` to `match side { Side::Bid => Direction::Up, Side::Ask => Direction::Down }`.

**Call sites (2):**

- `src/screen/dashboard/panel/ladder.rs:108`: `.update(raw_best_bid, true, update_t, max_int)`
- `src/screen/dashboard/panel/ladder.rs:110`: `.update(raw_best_ask, false, update_t, max_int)`

## Architecture Patterns

### Enum Placement Convention

Follow the existing codebase pattern: define enums close to the types that use them.

| Enum               | Location                                                   | Reason                      |
| ------------------ | ---------------------------------------------------------- | --------------------------- |
| (reuse `SizeUnit`) | `exchange/src/unit/qty.rs`                                 | Already exists              |
| `EmaAction`        | `data/src/conditional_ema.rs`                              | Only used by ConditionalEma |
| `Side`             | `data/src/panel/ladder.rs` (promote visibility)            | Shared by ladder + heatmap  |
| `GapFillProgress`  | `src/chart/kline/odb_core.rs` or `src/screen/dashboard.rs` | Used in ODB gap-fill flow   |

### Enum Design Pattern

```rust
// Pattern: two-variant enum with no data, Copy + Clone + Debug
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapFillProgress {
    Streaming,
    Complete,
}
```

For serialized structs, add serde derives + backward compat:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}
```

### Anti-Patterns to Avoid

- **Adding `impl From<bool>` or `.into()` for enums:** Defeats the purpose. Callers should use explicit variants.
- **Renaming without updating message variants:** If a bool propagates through a Message enum field (like `TradesReceived`), the enum must propagate there too.
- **Breaking serde compatibility on persisted structs:** Check whether structs are in the saved-state.json serialization path before changing field types.

## Don't Hand-Roll

| Problem             | Don't Build         | Use Instead                                         | Why                                            |
| ------------------- | ------------------- | --------------------------------------------------- | ---------------------------------------------- |
| Bid/Ask distinction | New enum per module | Shared `Side` enum (already in ladder.rs)           | Same concept used in ladder + heatmap          |
| Base/Quote currency | New enum            | Existing `SizeUnit` from `exchange/src/unit/qty.rs` | Already exists, already used at all call sites |

## Common Pitfalls

### Pitfall 1: Serde Backward Compatibility

**What goes wrong:** Changing `is_bid: bool` to `side: Side` on a serialized struct breaks deserialization of existing saved state.
**Why it happens:** `saved-state.json` may contain serialized heatmap/coalescing data.
**How to avoid:** Verify which structs appear in the saved-state serialization path. `OrderRun` and `CoalescingRun` are runtime depth data (ephemeral, rebuilt from WebSocket), NOT persisted in saved-state.json. The `HeatmapStudy` enum IS serialized but uses no bool fields. Safe to change without serde migration.
**Warning signs:** `#[derive(Serialize, Deserialize)]` on the struct.

### Pitfall 2: Forgetting Message Variant Propagation

**What goes wrong:** Changing `insert_raw_trades` parameter but not the `Message::TradesReceived` field that carries the same bool.
**Why it happens:** The bool originates in a message enum, flows through `dashboard.rs::update()`, and reaches the function.
**How to avoid:** Trace the bool from its origin (Message variant) through dispatch to the final function. Update all points in the chain.
**Warning signs:** Mismatched types between message field and function parameter.

### Pitfall 3: The `round_to_side_step` Bool

**What goes wrong:** Temptation to also refactor `Price::round_to_side_step(self, is_sell_or_bid: bool, step)` which has ~20 call sites across the codebase.
**Why it happens:** It uses a bool parameter with confusing semantics (the bool means "floor" for bids/sells, "ceil" for asks).
**How to avoid:** `round_to_side_step` is NOT one of the 5 identified targets. It has ~20 call sites across exchange adapter code and would significantly expand scope. Stay within the 5 identified functions only.
**Warning signs:** Expanding scope beyond QUAL-01's explicit list of 5 files.

## Code Examples

### Target 1: adapter.rs -- SizeUnit reuse

```rust
// BEFORE (exchange/src/adapter.rs)
pub fn qty_in_quote_value<T>(&self, qty: T, price: Price, size_in_quote_ccy: bool) -> f32
// Call site:
let size_in_quote_ccy = volume_size_unit() == SizeUnit::Quote;
market_type.qty_in_quote_value(qty, price, size_in_quote_ccy)

// AFTER
pub fn qty_in_quote_value<T>(&self, qty: T, price: Price, unit: SizeUnit) -> f32
where T: Into<f32> {
    let qty = qty.into();
    match self {
        MarketKind::InversePerps => qty,
        _ => match unit {
            SizeUnit::Quote => qty,
            SizeUnit::Base => price.to_f32() * qty,
        },
    }
}
// Call site:
market_type.qty_in_quote_value(qty, price, volume_size_unit())
```

### Target 4: odb_core.rs -- GapFillProgress

```rust
// BEFORE
pub fn insert_raw_trades(&mut self, raw_trades: Vec<Trade>, is_batches_done: bool)
// Call site in dashboard.rs:
c.insert_raw_trades(trades.to_owned(), is_batches_done);

// AFTER
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapFillProgress {
    Streaming,
    Complete,
}

pub fn insert_raw_trades(&mut self, raw_trades: Vec<Trade>, progress: GapFillProgress)
// Internal: if progress == GapFillProgress::Complete { ... }
```

### Target 5: ladder.rs -- Side reuse

```rust
// BEFORE (data/src/panel/ladder.rs)
pub fn update(&mut self, current_best: Option<Price>, is_bid: bool, now_ms: u64, max_interval: Duration)
// Call site:
self.bids.chase.update(raw_best_bid, true, update_t, max_int);
self.asks.chase.update(raw_best_ask, false, update_t, max_int);

// AFTER
pub fn update(&mut self, current_best: Option<Price>, side: Side, now_ms: u64, max_interval: Duration)
// Call site:
self.bids.chase.update(raw_best_bid, Side::Bid, update_t, max_int);
self.asks.chase.update(raw_best_ask, Side::Ask, update_t, max_int);
```

## Validation Architecture

### Test Framework

| Property           | Value                                       |
| ------------------ | ------------------------------------------- |
| Framework          | cargo clippy + cargo build (Rust compiler)  |
| Config file        | `clippy.toml`, `Cargo.toml`                 |
| Quick run command  | `cargo clippy --all-targets -- -D warnings` |
| Full suite command | `mise run lint`                             |

### Phase Requirements -> Test Map

| Req ID  | Behavior                         | Test Type            | Automated Command                                   | File Exists?   |
| ------- | -------------------------------- | -------------------- | --------------------------------------------------- | -------------- |
| QUAL-01 | 5 bool flags replaced with enums | compilation          | `cargo clippy --all-targets -- -D warnings`         | N/A (compiler) |
| VER-01  | clippy clean                     | compilation          | `cargo clippy --all-targets -- -D warnings`         | N/A (compiler) |
| VER-02  | Zero behavior change             | compilation + manual | `cargo build` (type system guarantees 1:1 mapping)  | N/A            |
| VER-03  | No new unsafe                    | grep audit           | `grep -r "unsafe" --include="*.rs" <changed files>` | N/A            |

### Sampling Rate

- **Per task commit:** `cargo clippy --all-targets -- -D warnings`
- **Per wave merge:** `mise run lint`
- **Phase gate:** Full `mise run lint` green before `/gsd:verify-work`

### Wave 0 Gaps

None -- existing Rust compiler and clippy infrastructure covers all phase requirements. No test framework setup needed.

## Sources

### Primary (HIGH confidence)

- Direct codebase analysis of all 5 target files and their call sites
- `exchange/src/adapter.rs:237` -- `qty_in_quote_value` signature and body
- `data/src/conditional_ema.rs:22` -- `ConditionalEma::update` signature and sole call site
- `data/src/chart/heatmap.rs:118-198` -- `OrderRun`, `process_side`, `update_price_level`
- `src/chart/kline/odb_core.rs:1259` -- `insert_raw_trades` signature and call chain
- `data/src/panel/ladder.rs:218` -- `ChaseTracker::update` signature and call sites
- `exchange/src/unit/qty.rs:16` -- existing `SizeUnit` enum
- `data/src/panel/ladder.rs:43` -- existing `Side` enum

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH -- pure Rust refactoring, no external dependencies
- Architecture: HIGH -- enum patterns well-established in codebase, existing precedents
- Pitfalls: HIGH -- serde compatibility verified (runtime-only structs), call sites exhaustively enumerated

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable -- pure refactoring, no dependency on external versions)

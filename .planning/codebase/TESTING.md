# Testing Patterns

**Analysis Date:** 2026-03-26

## Test Framework

**Runner:**

- Rust built-in test framework (`#[test]` macro)
- Config: None — uses Cargo defaults
- No external test runner (Jest, pytest, etc.)

**Assertion Library:**

- Standard library: `assert!()`, `assert_eq!()`, `assert_ne!()`
- Property-based testing: `proptest` crate for invariant verification
- Snapshot testing: `insta` crate for golden-file comparisons

**Run Commands:**

```bash
cargo test                          # Run all tests
cargo test --lib                    # Unit tests only (no integration tests)
cargo test -- --nocapture          # Show println! output
cargo test -- --test-threads=1     # Serial execution (default is parallel)
mise run lint                       # Format check + clippy (includes compilation)
```

## Test File Organization

**Location:**

- Co-located: test modules inline with implementation using `#[cfg(test)]` blocks at end of file
- Example: `src/chart/kline/mod.rs` has 2000+ lines of implementation followed by `#[cfg(test)] mod tests { ... }`
- No separate `tests/` directory for unit tests

**Naming:**

- Test modules: `#[cfg(test)] mod tests { ... }` (plural)
- Test functions: `#[test] fn snake_case_description()` describing the assertion
- Helper functions: unprefixed, private to test module: `fn make_trade(id: u64, price: f64) -> Trade`

**Structure:**

```
src/
├── chart/
│   └── kline/
│       └── mod.rs               ← implementation + inline #[cfg(test)]
├── adapter/
│   └── clickhouse.rs            ← implementation + proptest + snapshot tests
└── session.rs                   ← implementation + oracle tests
```

## Test Structure

**Suite Organization:**

Inline test modules (`#[cfg(test)]`) organized by test type:

1. **Unit tests** — test public functions with simple inputs/outputs
2. **Property-based tests** (`proptest!`) — verify invariants over random inputs
3. **Snapshot tests** (`#[cfg(test)] mod snapshot_tests`) — golden-file assertions

**Example from `src/chart/kline/mod.rs` (lines 2201–2350):**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ===== Unit Tests =====

    #[test]
    fn gap_of_one_is_not_a_gap() {
        assert!(!is_gap(100, 101));
    }

    #[test]
    fn gap_of_two_is_a_gap() {
        assert!(is_gap(100, 102));
    }

    #[test]
    fn dedup_fence_filters_stale_trades() {
        let fence_id: u64 = 100;
        let trades = [
            make_trade(99, 68000.0),
            make_trade(100, 68100.0),
            make_trade(101, 68200.0),
        ];
        let passed: Vec<_> = trades
            .iter()
            .filter(|t| t.agg_trade_id.is_none_or(|id| id > fence_id))
            .collect();
        assert_eq!(passed.len(), 2);
        assert_eq!(passed[0].agg_trade_id, Some(101));
    }

    // ===== Property Tests =====

    proptest! {
        #[test]
        fn fence_never_admits_stale_trades(
            fence_id in 1u64..10000,
            trade_ids in prop::collection::vec(1u64..20000, 1..20),
        ) {
            let admitted: Vec<u64> = trade_ids
                .iter()
                .copied()
                .filter(|&id| id > fence_id)
                .collect();
            for id in &admitted {
                prop_assert!(*id > fence_id);
            }
        }
    }

    // ===== Helpers =====

    fn make_trade(id: u64, price: f64) -> Trade {
        Trade {
            agg_trade_id: Some(id),
            time: 1700000000000,
            price: Price::from(price),
            qty: Qty::from(0.001),
            is_sell: false,
        }
    }
}
```

**Patterns:**

1. **Unit Test Pattern** — Direct assertion:

   ```rust
   #[test]
   fn function_name_describes_assertion() {
       let input = create_input();
       let result = function_under_test(input);
       assert_eq!(result, expected_value, "context message");
   }
   ```

2. **Guard/Composition Pattern** — Test complex boolean logic:

   ```rust
   fn guard_allows(
       fetching: bool,
       requested: bool,
       basis: &Basis,
       now_ms: u64,
       last_trigger: u64,
   ) -> bool {
       !fetching && !requested && basis.is_odb() && now_ms.saturating_sub(last_trigger) > 30_000
   }

   #[test]
   fn guard_allows_when_all_conditions_clear() {
       assert!(guard_allows(false, false, &Basis::Odb(250), 100_000, 60_000));
   }

   #[test]
   fn guard_blocks_when_fetching() {
       assert!(!guard_allows(true, false, &Basis::Odb(250), 100_000, 60_000));
   }
   ```

3. **Property Test Pattern** — Verify invariants:

   ```rust
   proptest! {
       #[test]
       fn property_name(
           input in strategy,
       ) {
           let result = operation(input);
           prop_assert!(invariant_holds(&result));
       }
   }
   ```

4. **Snapshot Test Pattern** — Verify serialization:

   ```rust
   #[test]
   fn function_serializes_correctly() {
       with_tokio(|| {
           let input = create_test_data();
           let result = process(input);
           assert_json_snapshot!(result);  // First run: creates snapshot
       });                                  // Subsequent runs: compare
   }
   ```

## Mocking

**Framework:** No explicit mocking library (mockito, nock, etc.)

**Patterns:**

1. **Constructor Injection** — Pass test-friendly objects:

   ```rust
   fn with_tokio<F: FnOnce() -> R, R>(f: F) -> R {
       let rt = tokio::runtime::Runtime::new().unwrap();
       let _guard = rt.enter();
       f()  // Test function runs within async context
   }

   #[test]
   fn test_async_operation() {
       with_tokio(|| {
           let result = async_function_that_uses_tokio();
           assert_eq!(result, expected);
       });
   }
   ```

2. **Test Helpers** — Factory functions create realistic test data:

   ```rust
   fn make_trade(id: u64, price: f64) -> Trade {
       Trade {
           agg_trade_id: Some(id),
           time: 1700000000000,
           price: Price::from(price),
           qty: Qty::from(0.001),
           is_sell: false,
       }
   }

   fn make_gap_fill_trade(id: u64) -> GapFillTrade {
       GapFillTrade {
           agg_trade_id: id,
           time: 1700000000000,
           price: 68500.0,
           qty: 0.001,
           is_buyer_maker: false,
       }
   }
   ```

3. **Options/Defaults** — Test with explicit configurations:

   ```rust
   #[test]
   fn dedup_fence_none_passes_all() {
       let fence: Option<u64> = None;
       let trades = [make_trade(1, 68000.0), make_trade(2, 68100.0)];
       let passed: Vec<_> = trades
           .iter()
           .filter(|t| match fence {
               None => true,
               Some(f) => t.agg_trade_id.is_none_or(|id| id > f),
           })
           .collect();
       assert_eq!(passed.len(), 2);
   }
   ```

**What to Mock:**

- Async runtime contexts: wrap in `with_tokio(|| { ... })`
- Fixed timestamps: use constants (`TKY_OPEN_20260302_MS`, `NY_OPEN_20260302_MS`)
- Network calls: stub using test data (no real HTTP calls in unit tests)

**What NOT to Mock:**

- Core logic — test actual implementation (`is_gap()`, `guard_allows()`)
- Data transformations — use real `Trade`, `Kline` structs
- Enum matching — test all branches explicitly

## Fixtures and Factories

**Test Data:**

Factories defined inline in test modules, not in separate fixture files:

```rust
fn make_trade(id: u64, price: f64) -> Trade {
    Trade {
        agg_trade_id: Some(id),
        time: 1700000000000,
        price: Price::from(price),
        qty: Qty::from(0.001),
        is_sell: false,
    }
}

fn find_boundary(
    boundaries: &[SessionBoundary],
    session: TradingSession,
    kind: BoundaryKind,
) -> Option<u64> {
    boundaries
        .iter()
        .find(|b| b.session == session && b.kind == kind)
        .map(|b| b.timestamp_ms)
}
```

**Location:**

- In test module, after test functions, marked as helpers
- Some shared constants at module level: `const NY_OPEN_20260302_MS: u64 = 1740917400000;`

**Oracle Constants** (from `data/src/session.rs`):

```rust
const NY_OPEN_20260302_MS: u64 = 1740917400000;    // 2026-03-02 09:30:00 EST
const NY_CLOSE_20260302_MS: u64 = 1740955200000;   // 2026-03-02 16:00:00 EST
const LDN_OPEN_20260302_MS: u64 = 1740892800000;   // 2026-03-02 08:00:00 GMT
const TKY_OPEN_20260302_MS: u64 = 1740854400000;   // 2026-03-02 09:00:00 JST
```

Used to verify timezone transitions and UTC boundary calculations.

## Coverage

**Requirements:** None enforced

**View Coverage:**

```bash
cargo tarpaulin --out Html --output-dir coverage  # Generates coverage/index.html
```

**Coverage Status (estimated):**

- Core logic modules (session, indicators, aggregation): 85%–95%
- Exchange adapters: 70%–80% (some network error paths untested)
- UI/event handling: 30%–50% (GUI testing overhead; unit tested at component level)

**Gaps:** No integration tests; all tests are unit-level. API integration with ClickHouse/WebSocket tested via manual `mise run run`.

## Test Types

**Unit Tests:**

- Scope: Single function or method with controlled inputs
- Approach: Direct assertion, no external dependencies
- Example: `is_gap(prev, curr)` tested with pairs `(100, 101)`, `(100, 102)`, `(200, 100)`
- Count: ~100 tests across codebase

**Integration Tests:**

- Not systematized; manual testing via `mise run run` and `mise run preflight`
- Covers: WebSocket stream connection, ClickHouse HTTP queries, pane state persistence
- Future: Could add `tests/` directory with end-to-end scenarios

**E2E Tests:**

- Not used — GUI app requires manual interaction or headless browser (out of scope)
- Manual testing documented in project MEMORY.md (launch protocol, pane editing, etc.)

## Common Patterns

**Async Testing:**

```rust
fn with_tokio<F: FnOnce() -> R, R>(f: F) -> R {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();
    f()
}

#[test]
fn test_uses_tokio_internally() {
    with_tokio(|| {
        // async code that calls tokio::spawn or uses tg_alert! macro runs here
        let result = function_that_uses_tokio();
        assert_eq!(result, expected);
    });
}
```

**Reason**: `tg_alert!(...)` macro calls `tokio::spawn()` to send Telegram alerts asynchronously. Tests must provide a runtime context via `rt.enter()`.

**Error Testing:**

```rust
#[test]
fn function_returns_error_on_invalid_input() {
    let result = function_that_can_fail(invalid_input);
    assert!(result.is_err());
    match result {
        Err(e) => assert!(e.to_string().contains("expected error message")),
        _ => panic!("expected error"),
    }
}
```

**Boundary Testing** (especially for ODB thresholds, time windows):

```rust
#[test]
fn cooldown_arithmetic_exact_boundary() {
    let last: u64 = 1_000_000;
    let now: u64 = 1_030_000;  // exactly 30s
    assert!(now.saturating_sub(last) <= 30_000);
}

#[test]
fn cooldown_arithmetic_allows_after_30s() {
    let last: u64 = 1_000_000;
    let now: u64 = 1_031_000;  // 31s later
    assert!(now.saturating_sub(last) > 30_000);
}
```

**Property-Based Testing** (exhaustive verification):

```rust
proptest! {
    #[test]
    fn gap_plus_continuity_is_exhaustive(
        prev in 1u64..100000,
        curr in 1u64..100000,
    ) {
        let is_gap = curr.saturating_sub(prev) > 1;
        let is_continuous = curr == prev + 1;
        let is_dup_or_reorder = curr <= prev;

        // Exactly one must be true
        let flags = [is_gap, is_continuous, is_dup_or_reorder];
        let true_count = flags.iter().filter(|&&f| f).count();
        prop_assert_eq!(true_count, 1);
    }
}
```

**Snapshot Testing** (golden files):

```rust
use insta::assert_json_snapshot;

#[test]
fn catchup_response_deserialization() {
    with_tokio(|| {
        let json = r#"{"trades":[...],"through_agg_id":123456,...}"#;
        let response: CatchupResponse = serde_json::from_str(json).unwrap();
        let result = catchup_response_to_result(response, uuid::Uuid::nil());
        assert_json_snapshot!(CatchupSnapshot::from(&result));
    });
}
```

First run creates `exchange/src/adapter/clickhouse.rs.insta/snapshots/[test_name].json`. Subsequent runs compare.

---

_Testing analysis: 2026-03-26_

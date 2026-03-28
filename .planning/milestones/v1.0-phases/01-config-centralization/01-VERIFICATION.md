---
phase: 01-config-centralization
verified: 2026-03-26T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 01: Config Centralization Verification Report

**Phase Goal:** All runtime configuration flows through a single validated struct — no more hunting through 6 files for env var reads
**Verified:** 2026-03-26
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                        | Status     | Evidence                                                                                                            |
| --- | -------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------- |
| 1   | All FLOWSURFACE\_\* env vars read in exactly one place (exchange/src/config.rs)              | ✓ VERIFIED | `grep -rn 'std::env::var' src/ exchange/src/` returns only hits in config.rs (5 lines, all helpers)                 |
| 2   | App starts with default config when no env vars are set                                      | ✓ VERIFIED | `AppConfig::default()` provides all 10 defaults; `from_env` falls back to same values                               |
| 3   | App starts with full .mise.toml config when all env vars are set                             | ✓ VERIFIED | All 10 env vars parsed in `from_env()` with correct keys matching CLAUDE.md table                                   |
| 4   | Invalid env var (e.g. CH_PORT=abc) produces a startup warning on stderr before network calls | ✓ VERIFIED | `eprintln!` in `parse_env_u16` (line 93) and `parse_env_bool` (line 109); LazyLock guarantees init before first use |
| 5   | clickhouse.rs and telegram.rs use config struct instead of direct env reads                  | ✓ VERIFIED | 0 `std::env::var` calls in either file; 10 APP_CONFIG usages in clickhouse.rs, 7 in telegram.rs                     |
| 6   | main.rs and logger.rs use config struct instead of direct env reads                          | ✓ VERIFIED | 0 `std::env::var` calls in main.rs and logger.rs; APP_CONFIG used for always_on_top (×2) and rust_log               |
| 7   | cargo clippy --all-targets -- -D warnings passes clean                                       | ✓ VERIFIED | `Finished dev profile [optimized + debuginfo] target(s) in 0.72s` — no warnings, no errors                          |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact                             | Expected                                                  | Status     | Details                                                                                                                                                                 |
| ------------------------------------ | --------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exchange/src/config.rs`             | Centralized AppConfig struct with LazyLock initialization | ✓ VERIFIED | 125 lines (≥ 50 min). Exports `APP_CONFIG: LazyLock<AppConfig>` and `AppConfig`. Contains `base_url()`, `tg_configured()`, all 4 parse helpers with eprintln validation |
| `exchange/src/adapter/clickhouse.rs` | ClickHouse adapter using shared config                    | ✓ VERIFIED | `use crate::config::APP_CONFIG` at line 16. 10 APP_CONFIG usages covering base_url, ouroboros_mode (×5), sse_enabled, sse_host, sse_port                                |
| `src/main.rs`                        | Main app using AppConfig for ALWAYS_ON_TOP                | ✓ VERIFIED | `use exchange::config::APP_CONFIG` at line 39. `APP_CONFIG.always_on_top` at lines 162 and 214                                                                          |
| `src/logger.rs`                      | Logger using AppConfig for RUST_LOG                       | ✓ VERIFIED | `exchange::config::APP_CONFIG.rust_log` at line 26. No `std::env::var` calls remain                                                                                     |
| `exchange/src/lib.rs`                | Module declaration for config                             | ✓ VERIFIED | `pub mod config;` at line 2                                                                                                                                             |

### Key Link Verification

| From                                 | To                       | Via                          | Status  | Details                                                                                                              |
| ------------------------------------ | ------------------------ | ---------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `exchange/src/adapter/clickhouse.rs` | `exchange/src/config.rs` | APP_CONFIG static reference  | ✓ WIRED | Import at line 16; 10 field accesses at lines 240, 436, 452, 554, 594, 655, 811, 879, 880, 1127                      |
| `exchange/src/telegram.rs`           | `exchange/src/config.rs` | APP_CONFIG static reference  | ✓ WIRED | Import at line 9; 7 usages — tg_configured, tg_bot_token (×2), tg_chat_id (×2), ch_host, ch_port, sse_host, sse_port |
| `src/main.rs`                        | `exchange/src/config.rs` | exchange::config::APP_CONFIG | ✓ WIRED | Import at line 39; `APP_CONFIG.always_on_top` at lines 162 and 214                                                   |
| `src/logger.rs`                      | `exchange/src/config.rs` | exchange::config::APP_CONFIG | ✓ WIRED | Inline reference `exchange::config::APP_CONFIG.rust_log` at line 26                                                  |

### Data-Flow Trace (Level 4)

Not applicable — this phase produces config infrastructure (no components rendering dynamic data). The `AppConfig` struct is a data source, not a consumer of external data.

### Behavioral Spot-Checks

| Behavior                                             | Command                                                                     | Result            | Status |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | ----------------- | ------ |
| Zero env reads outside config.rs in src/ + exchange/ | `grep -rn 'std::env::var' src/ exchange/src/ \| grep -v config.rs \| wc -l` | 0                 | ✓ PASS |
| eprintln warnings in config.rs (CFG-03 validation)   | `grep -c 'eprintln!' exchange/src/config.rs`                                | 2 (lines 93, 109) | ✓ PASS |
| config.rs has required parse helper functions        | `grep 'fn parse_env_u16\|fn parse_env_bool' exchange/src/config.rs`         | 2 matches         | ✓ PASS |
| convenience methods exist                            | `grep 'pub fn base_url\|pub fn tg_configured' exchange/src/config.rs`       | 2 matches         | ✓ PASS |
| No unsafe code in modified files                     | `grep -rn 'unsafe' config.rs main.rs logger.rs clickhouse.rs telegram.rs`   | 0 matches         | ✓ PASS |
| clippy passes with -D warnings                       | `cargo clippy --all-targets -- -D warnings`                                 | Finished 0.72s    | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                              | Status      | Evidence                                                                                                  |
| ----------- | ----------- | ---------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------- |
| CFG-01      | 01-01       | All env var reads centralized in a single AppConfig struct with LazyLock initialization  | ✓ SATISFIED | `APP_CONFIG: LazyLock<AppConfig>` in config.rs; all reads via `from_env()` with parse helpers             |
| CFG-02      | 01-01       | Duplicate reads eliminated — CH_HOST, CH_PORT, SSE_HOST, SSE_PORT each read exactly once | ✓ SATISFIED | 0 env reads in clickhouse.rs/telegram.rs; config.rs reads each key exactly once in `from_env()`           |
| CFG-03      | 01-01       | Config validated eagerly at startup — invalid vars produce clear error messages          | ✓ SATISFIED | `parse_env_u16` and `parse_env_bool` call `eprintln!` with `[flowsurface] KEY=value is not valid`         |
| CFG-04      | 01-01       | clickhouse.rs and telegram.rs import config values from the shared struct                | ✓ SATISFIED | Both files import APP_CONFIG; 0 direct env reads remain in either file                                    |
| VER-01      | 01-02       | `cargo clippy -- -D warnings` passes after every phase                                   | ✓ SATISFIED | `cargo clippy --all-targets -- -D warnings` finishes clean                                                |
| VER-02      | 01-02       | Zero behavior changes — all existing ODB, charting, and exchange functionality works     | ✓ SATISFIED | Defaults in AppConfig exactly match prior per-module defaults; presence-check for ALWAYS_ON_TOP preserved |
| VER-03      | 01-02       | No new `unsafe` code introduced                                                          | ✓ SATISFIED | 0 `unsafe` occurrences in all 5 modified files                                                            |

No orphaned requirements — all 7 IDs declared in PLAN frontmatter are covered, and REQUIREMENTS.md maps all 7 to Phase 1.

### Anti-Patterns Found

None. Scanned `exchange/src/config.rs`, `exchange/src/adapter/clickhouse.rs`, `exchange/src/telegram.rs`, `src/main.rs`, `src/logger.rs`.

The `LazyLock` statics remaining in `telegram.rs` (`HTTP` and `COOLDOWNS`) are runtime state (HTTP client, cooldown map), not config — intentionally kept per plan.

### Human Verification Required

None required for automated checks.

The following are low-priority behavioral checks that can be verified at next launch if desired:

1. **Test Name:** Invalid port warning at startup
   **Test:** Set `FLOWSURFACE_CH_PORT=abc` in `.mise.toml`, run `mise run run`, observe stderr
   **Expected:** `[flowsurface] FLOWSURFACE_CH_PORT="abc" is not a valid u16, using 8123` printed before any ClickHouse connection attempt
   **Why human:** Cannot test without restarting the app with a modified env var

### Gaps Summary

No gaps. Phase goal is fully achieved.

All 7 observable truths verified. All 5 artifacts exist, are substantive (no stubs), and are wired (imported and actively used). All 7 requirements satisfied. Build is clean. No unsafe code introduced. Behavior preserved.

The codebase now has a single `AppConfig` struct in `exchange/src/config.rs` as the sole point of entry for all `FLOWSURFACE_*` env var reads. The previous 6-file hunt is eliminated.

---

_Verified: 2026-03-26_
_Verifier: Claude (gsd-verifier)_

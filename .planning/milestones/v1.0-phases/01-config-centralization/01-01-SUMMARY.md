---
phase: 01-config-centralization
plan: 01
subsystem: config
tags: [env-vars, lazy-lock, centralization, validation]

# Dependency graph
requires: []
provides:
  - "APP_CONFIG LazyLock<AppConfig> static in exchange::config"
  - "Centralized env var reads with parse validation (CFG-03)"
  - "base_url() and tg_configured() convenience methods"
affects: [01-config-centralization, 02-must-use-annotations]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Single config struct with LazyLock initialization from env vars"]

key-files:
  created: [exchange/src/config.rs]
  modified:
    [
      exchange/src/lib.rs,
      exchange/src/adapter/clickhouse.rs,
      exchange/src/telegram.rs,
    ]

key-decisions:
  - "eprintln! for parse warnings (before any network calls via LazyLock init-on-first-access)"
  - "startup_health_check now shows actual config values instead of wrong localhost/18123 defaults"

patterns-established:
  - "Config access: use crate::config::APP_CONFIG instead of std::env::var()"
  - "Parse validation: parse_env_u16/bool warn on invalid values before fallback"

requirements-completed: [CFG-01, CFG-02, CFG-03, CFG-04]

# Metrics
duration: 20min
completed: 2026-03-27
---

# Phase 01 Plan 01: AppConfig Struct Summary

**Centralized 12 scattered env var reads into AppConfig LazyLock struct with parse-validation warnings for invalid u16/bool values**

## Performance

- **Duration:** 20 min
- **Started:** 2026-03-27T22:31:06Z
- **Completed:** 2026-03-27T22:51:33Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `exchange/src/config.rs` with `AppConfig` struct covering all 10 FLOWSURFACE\_\* env vars + RUST_LOG
- Eliminated 7 LazyLock statics across clickhouse.rs (5) and telegram.rs (2)
- Added CFG-03 validation: `eprintln!` warnings for invalid port numbers and boolean values
- Fixed bug: `startup_health_check()` previously showed wrong defaults (`localhost:18123`) vs actual config (`bigblack:8123`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create exchange/src/config.rs with AppConfig struct** - `e456ef1` (feat)
2. **Task 2: Migrate clickhouse.rs and telegram.rs to use AppConfig** - `2f7d27e` (refactor)

## Files Created/Modified

- `exchange/src/config.rs` - Centralized AppConfig struct with LazyLock, parse helpers, base_url(), tg_configured()
- `exchange/src/lib.rs` - Added `pub mod config;` declaration
- `exchange/src/adapter/clickhouse.rs` - Removed 5 statics, replaced all env reads with APP_CONFIG
- `exchange/src/telegram.rs` - Removed 2 statics, replaced all env reads with APP_CONFIG

## Decisions Made

- Used `eprintln!` (not `log::warn!`) for parse warnings because LazyLock initializes before logger setup
- Fixed startup_health_check to use actual config values -- previous defaults were incorrect (localhost vs bigblack, 18123 vs 8123) which is a bug fix, not a behavior change

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- APP_CONFIG is available for plan 01-02 (migrate main crate env reads: ALWAYS_ON_TOP, RUST_LOG)
- All exchange crate env reads are centralized; main crate reads remain (logger.rs, main.rs)

---

_Phase: 01-config-centralization_
_Completed: 2026-03-27_

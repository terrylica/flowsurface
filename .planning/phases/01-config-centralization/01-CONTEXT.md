# Phase 1: Config Centralization - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

All runtime configuration flows through a single validated struct — no more hunting through 6 files for env var reads. Creates `exchange/src/config.rs` with a `LazyLock<AppConfig>` struct that centralizes all `FLOWSURFACE_*` env var reads.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion

All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

Key constraints from research:

- LazyLock eager init pattern to prevent timing issues (PITFALLS.md)
- Config struct with Default trait for fail-fast startup (SSoT principle)
- Keep defaults consistent with current .mise.toml values
- `pub(crate)` visibility — config is internal to exchange crate, consumed by clickhouse.rs and telegram.rs

</decisions>

<code_context>

## Existing Code Insights

### Current Env Var Locations

- `exchange/src/adapter/clickhouse.rs:238-250` — CLICKHOUSE_HOST, CLICKHOUSE_PORT, OUROBOROS_MODE (LazyLock statics)
- `exchange/src/adapter/clickhouse.rs:818-834` — SSE_HOST, SSE_PORT, SSE_ENABLED
- `exchange/src/telegram.rs:12-15,114-117` — TG_BOT_TOKEN, TG_CHAT_ID, CH_HOST, CH_PORT, SSE_HOST, SSE_PORT (duplicates!)
- `data/src/lib.rs:127` — FLOWSURFACE_DATA_PATH
- `data/src/telemetry.rs:28` — FLOWSURFACE_TELEMETRY
- `src/main.rs:161,213` — FLOWSURFACE_ALWAYS_ON_TOP (2 reads)
- `src/logger.rs:26` — RUST_LOG

### Duplicate Reads (4 vars)

- CH_HOST: clickhouse.rs + telegram.rs
- CH_PORT: clickhouse.rs + telegram.rs
- SSE_HOST: clickhouse.rs + telegram.rs
- SSE_PORT: clickhouse.rs + telegram.rs

### Established Patterns

- `LazyLock<T>` used throughout for one-time init
- `std::env::var().unwrap_or_else()` for defaults
- `OnceLock` for values set post-init (ODB_SYMBOLS, RUNTIME_PROXY_CFG)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria.

</specifics>

<deferred>
## Deferred Ideas

None — discuss phase skipped.

</deferred>

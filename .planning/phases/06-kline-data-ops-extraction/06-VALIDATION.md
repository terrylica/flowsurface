---
phase: 6
slug: kline-data-ops-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 6 — Validation Strategy

## Test Infrastructure

| Property               | Value                                       |
| ---------------------- | ------------------------------------------- |
| **Framework**          | cargo clippy + cargo check                  |
| **Quick run command**  | `cargo check 2>&1 \| tail -5`               |
| **Full suite command** | `cargo clippy --all-targets -- -D warnings` |
| **Estimated runtime**  | ~30 seconds                                 |

## Sampling Rate

- **After every task commit:** Run `cargo check`
- **After every plan wave:** Run full clippy suite
- **Max feedback latency:** 30 seconds

## Manual-Only Verifications

| Behavior             | Requirement | Why Manual           | Test Instructions             |
| -------------------- | ----------- | -------------------- | ----------------------------- |
| Historical bars load | VER-02      | Requires running app | Open pane, verify bars appear |
| Live trades flow     | VER-02      | Requires running app | Watch forming bar update      |

**Approval:** pending

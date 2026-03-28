---
phase: 3
slug: bool-to-enum-cleanup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property               | Value                                       |
| ---------------------- | ------------------------------------------- |
| **Framework**          | cargo clippy + cargo check (Rust compiler)  |
| **Config file**        | `clippy.toml`                               |
| **Quick run command**  | `cargo check 2>&1 \| tail -5`               |
| **Full suite command** | `cargo clippy --all-targets -- -D warnings` |
| **Estimated runtime**  | ~30 seconds                                 |

---

## Sampling Rate

- **After every task commit:** Run `cargo check 2>&1 | tail -5`
- **After every plan wave:** Run `cargo clippy --all-targets -- -D warnings`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID  | Plan | Wave | Requirement | Test Type | Automated Command                           | File Exists | Status     |
| -------- | ---- | ---- | ----------- | --------- | ------------------------------------------- | ----------- | ---------- |
| 03-01-01 | 01   | 1    | QUAL-01     | compiler  | `cargo clippy --all-targets -- -D warnings` | ✅          | ⬜ pending |

_Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky_

---

## Wave 0 Requirements

_Existing infrastructure covers all phase requirements. `cargo clippy` is already configured with `-D warnings`._

---

## Manual-Only Verifications

| Behavior                         | Requirement | Why Manual                | Test Instructions                                                                           |
| -------------------------------- | ----------- | ------------------------- | ------------------------------------------------------------------------------------------- |
| Call sites read as enum variants | QUAL-01     | Readability is subjective | Grep for `true` and `false` in target function call sites — should find zero bare bool args |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

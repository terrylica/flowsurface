---
phase: 2
slug: must-use-safety-net
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-27
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property               | Value                                               |
| ---------------------- | --------------------------------------------------- |
| **Framework**          | cargo clippy + cargo check (Rust compiler warnings) |
| **Config file**        | `clippy.toml`                                       |
| **Quick run command**  | `cargo check 2>&1 \| grep "unused.*must_use"`       |
| **Full suite command** | `cargo clippy --all-targets -- -D warnings`         |
| **Estimated runtime**  | ~30 seconds                                         |

---

## Sampling Rate

- **After every task commit:** Run `cargo check 2>&1 | grep "unused.*must_use"`
- **After every plan wave:** Run `cargo clippy --all-targets -- -D warnings`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID  | Plan | Wave | Requirement | Test Type | Automated Command                           | File Exists | Status     |
| -------- | ---- | ---- | ----------- | --------- | ------------------------------------------- | ----------- | ---------- |
| 02-01-01 | 01   | 1    | QUAL-02     | compiler  | `cargo clippy --all-targets -- -D warnings` | ✅          | ⬜ pending |
| 02-01-02 | 01   | 1    | VER-01      | compiler  | `cargo check 2>&1 \| grep must_use`         | ✅          | ⬜ pending |

_Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky_

---

## Wave 0 Requirements

_Existing infrastructure covers all phase requirements. `cargo clippy` is already configured with `-D warnings`._

---

## Manual-Only Verifications

| Behavior                          | Requirement | Why Manual                           | Test Instructions                                                                 |
| --------------------------------- | ----------- | ------------------------------------ | --------------------------------------------------------------------------------- |
| Intentional drop triggers warning | VER-03      | Requires writing throwaway test code | Add `fn _test() { pane_state.update(...); }` without `let _`, verify clippy warns |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

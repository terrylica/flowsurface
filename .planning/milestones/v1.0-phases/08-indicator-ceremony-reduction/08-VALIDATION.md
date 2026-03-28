---
phase: 8
slug: indicator-ceremony-reduction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 8 — Validation Strategy

## Test Infrastructure

| Property               | Value                                       |
| ---------------------- | ------------------------------------------- |
| **Framework**          | cargo clippy                                |
| **Quick run command**  | `cargo check 2>&1 \| tail -5`               |
| **Full suite command** | `cargo clippy --all-targets -- -D warnings` |
| **Estimated runtime**  | ~30 seconds                                 |

## Manual-Only Verifications

| Behavior           | Requirement | Why Manual                                 |
| ------------------ | ----------- | ------------------------------------------ |
| Checklist accuracy | QUAL-03     | Must verify documented steps match reality |

**Approval:** pending

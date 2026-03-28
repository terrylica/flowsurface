---
phase: 7
slug: kline-odb-lifecycle-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 7 — Validation Strategy

## Test Infrastructure

| Property               | Value                                                                       |
| ---------------------- | --------------------------------------------------------------------------- |
| **Framework**          | cargo clippy + wc -l                                                        |
| **Quick run command**  | `cargo check 2>&1 \| tail -5`                                               |
| **Full suite command** | `cargo clippy --all-targets -- -D warnings && wc -l src/chart/kline/mod.rs` |
| **Estimated runtime**  | ~30 seconds                                                                 |

## Manual-Only Verifications

| Behavior                                      | Requirement | Why Manual                |
| --------------------------------------------- | ----------- | ------------------------- |
| ODB pane loads + gap-fills + live forming bar | KLINE-02    | Requires app + ClickHouse |
| RefCell borrow discipline preserved           | VER-02      | Runtime check only        |

**Approval:** pending

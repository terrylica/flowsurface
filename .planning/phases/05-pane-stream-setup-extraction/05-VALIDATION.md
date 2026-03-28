---
phase: 5
slug: pane-stream-setup-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property               | Value                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------- |
| **Framework**          | cargo clippy + wc -l (LOC count)                                                      |
| **Config file**        | `clippy.toml`                                                                         |
| **Quick run command**  | `cargo check 2>&1 \| tail -5`                                                         |
| **Full suite command** | `cargo clippy --all-targets -- -D warnings && wc -l src/screen/dashboard/pane/mod.rs` |
| **Estimated runtime**  | ~30 seconds                                                                           |

---

## Sampling Rate

- **After every task commit:** Run `cargo check && wc -l src/screen/dashboard/pane/mod.rs`
- **After every plan wave:** Run full suite
- **Max feedback latency:** 30 seconds

---

## Manual-Only Verifications

| Behavior                          | Requirement | Why Manual                        | Test Instructions                                            |
| --------------------------------- | ----------- | --------------------------------- | ------------------------------------------------------------ |
| ODB pane loads with all 3 streams | PANE-02     | Requires running app + ClickHouse | Launch app, open ODB pane, verify bars + trades + depth load |
| Ticker switch refreshes streams   | VER-02      | Requires running app              | Switch ticker on live pane, verify data refreshes            |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

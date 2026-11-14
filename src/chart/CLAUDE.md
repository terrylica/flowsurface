# Chart Rendering

**Parent**: [/CLAUDE.md](/CLAUDE.md)

Chart canvas rendering, indicator integration, ODB bar interaction, and visual overlays. All files under `src/chart/`.

---

## Quick Reference

| Module          | File                     | Purpose                                    |
| --------------- | ------------------------ | ------------------------------------------ |
| Chart facade    | `src/chart.rs`           | Re-exports `ViewState`, `Message`, modules |
| Kline rendering | `kline/mod.rs`           | Candle/ODB/footprint draw, trade insertion |
| ODB core        | `kline/odb_core.rs`      | Gap-fill, dedup fence, forming bar         |
| ODB lifecycle   | `kline/odb_lifecycle.rs` | Sentinel audit, refetch triggers           |
| Rendering prims | `kline/rendering.rs`     | `draw_candle_dp`, `draw_diamond`, clusters |
| Bar selection   | `kline/bar_selection.rs` | Shift+Click range selection, stats overlay |
| Crosshair       | `kline/crosshair.rs`     | Tooltip, price/time labels                 |
| View state      | `kline/view_state.rs`    | Viewport: translation, scaling, cell_width |
| Scale           | `kline/scale.rs`         | Price/time axis rendering                  |
| Session lines   | `session.rs`             | NY/London/Tokyo dotted lines + strips      |
| Indicator trait | `indicator/kline.rs`     | `KlineIndicatorImpl` trait                 |
| Indicator impls | `indicator/kline/*.rs`   | Volume, RSI, OFI, TradeIntensity, ZigZag   |
| Heatmap chart   | `heatmap.rs`             | Depth heatmap (orderbook visualization)    |

---

## iced Canvas Architecture

**Four geometry layers** in `kline/mod.rs` (stacked in draw order):

| Layer       | Frame transforms?             | When cleared            |
| ----------- | ----------------------------- | ----------------------- |
| `main`      | translate → scale → translate | Panning, zoom, new bars |
| `watermark` | None (screen-space)           | Rarely                  |
| `legend`    | None (screen-space)           | Every cursor move       |
| `crosshair` | None (screen-space)           | Every cursor move       |

**Chart-space → screen-space formula** (canonical, from `keyboard_nav.rs`):

```
screen_x = (chart_x + translation.x) * scaling + bounds.width / 2
```

ODB: `chart_x = -(visual_idx * cell_width)`. Lower `visual_idx` = newer bar = higher screen_x.

### Hit Detection

| ❌ Anti-pattern                                                            | ✅ Correct pattern                                                                              |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Compute `screen_x` from formula; check `(cursor.x − screen_x).abs() < HIT` | `snap_x_to_index(cursor_pos.x, bounds_size, region)` → bar index; check `abs_diff(target) <= 1` |

Reason: `cursor.position_in(bounds)` and manual screen math can have subtle discrepancies. `snap_x_to_index` is canonical (same function used for crosshair + Shift+Click) and is guaranteed grid-consistent.

**Visual handle width should match hit zone**: if hit zone is ±1 bar, draw the handle `cell_width * scaling` px wide (one full bar on screen). Mismatched visual ↔ hit zones confuse users.

### Interior Mutability

`canvas::Program::update()` takes `&self`. Wrap mutable canvas state in `RefCell<T>`. Borrow immutably to extract values → drop borrow → `borrow_mut()` to update. Never hold an immutable borrow across a `borrow_mut()` call.

---

## Adding a New Indicator

**Standard subplot indicator (3 files: 2 modified + 1 new):**

1. **`data/src/chart/indicator.rs`** (enum + arrays + Display):
   - Add variant to `KlineIndicator` enum
   - Add to `FOR_SPOT` and/or `FOR_PERPS` arrays (increment array length constant)
   - Add `Display` match arm

2. **`src/chart/indicator/kline.rs`** (factory + module):
   - Add `pub mod <name>;` declaration
   - Add match arm to `make_indicator()` factory

3. **`src/chart/indicator/kline/<name>.rs`** (NEW FILE — implementation):
   - Implement `KlineIndicatorImpl` trait (see `rsi.rs` or `volume.rs` for reference)

**Extended ceremony (only for configurable or special-rendering indicators):**

- **Configurable params** (e.g., EMA period): also modify `data/src/chart/kline.rs` Config struct + update `make_indicator()` arm to use config
- **Main-canvas overlay** (e.g., ZigZag lines, anomaly diamonds): implement `draw_overlay()` on the trait — drawn after candles in main cache layer
- **Body recoloring** (e.g., heatmap thermal): implement `thermal_body_color()` on the trait — replaces candle body color per bar

No other files need touching. `EnumMap` derive auto-expands storage. Serde derive handles serialization.

---

## Extending ODB Support

When modifying ODB rendering or behavior, check **all** match arms for `Basis::Odb(_)`, `KlineChartKind::Odb`, and `ContentKind::OdbChart` across:

- `src/screen/dashboard/pane.rs` — pane streams (must include `OdbKline` + `Depth` + `Trades`)
- `src/screen/dashboard.rs` — event dispatch, pane switching
- `src/chart/kline.rs` — rendering, trade insertion
- `src/chart/heatmap.rs` — depth heatmap
- `src/modal/pane/stream.rs` — settings UI
- `src/modal/pane/settings.rs` — chart config
- `data/src/layout/pane.rs` — serialization

---

## Anomaly Fence (Adjusted Boxplot)

When the Trade Intensity Heatmap indicator has `anomaly_fence: true`, bars with anomalously low trade intensity (conformal p-value < α=0.05) get **direction-aware diamond markers**:

- **Bullish anomaly** → diamond above the bar's high wick
- **Bearish anomaly** → diamond below the bar's low wick
- **Color**: yellow (p ≈ 0.05, mild) → red (p → 0, extreme)
- **Size**: half_size 2–4px, scaling with severity

Implementation: `draw_overlay()` on `TradeIntensityHeatmapIndicator` (`indicator/kline/trade_intensity_heatmap.rs`). Uses `draw_diamond()` primitive from `kline/rendering.rs`. Candle body rendering (`draw_candle_dp`) is not involved — full separation of concerns.

---

## ODB Bar Range Selection

**File**: `kline/bar_selection.rs` — `BarSelectionState` (in `RefCell`) + `BrimSide` enum.

**UX**:

- Shift+Left Click: anchor → end → restart (3rd click resets to new anchor)
- Left Click on outermost bar of selection: drag that brim to relocate boundary
- `u64::MAX` sentinel from `snap_x_to_index` = forming-bar zone; ignore there

**Cache strategy**: selection highlight drawn in `crosshair` layer; stats in `legend` layer. Neither invalidates the heavy `main` (candles) cache during drag. Only `clear_crosshair()` + `legend.clear()` on `CursorMoved`.

**Stats overlay** (top-center, `legend` layer): `N bars` / `↑ up (%)` / `↓ down (%)`. Distance = `|end − anchor|` (0 = same bar, 1 = adjacent).

See [docs/audits/CLAUDE.md](../docs/audits/CLAUDE.md) for statistical audit of selection metrics.

---

## Common Errors (Chart-Specific)

| Error                                  | Cause                                                                 | Fix                                                                                                         |
| -------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Tiny dot candlesticks                  | Wrong cell_width/limit                                                | Check adaptive scaling in `kline/mod.rs`                                                                    |
| Crosshair panic                        | NaN in indicator data                                                 | Add NaN guard before rendering                                                                              |
| Intensity heatmap colors stop at K≈13  | `FetchRange::Kline(0,now)` hit `LIMIT 2000` instead of adaptive limit | Initial fetches must use `FetchRange::Kline(0, u64::MAX)` — see [exchange/CLAUDE.md](../exchange/CLAUDE.md) |
| Brim drag / hit detection misses       | Screen-space formula used for hit testing                             | Use `snap_x_to_index()` ± 1 bar; never compute screen positions manually                                    |
| Legend shows wrong day at day boundary | `prev_bar.close_time` used as open time                               | Use `TickAccumulation.open_time_ms` — see [data/CLAUDE.md](../data/CLAUDE.md)                               |

---

## Related

- [/CLAUDE.md](/CLAUDE.md) — Project hub
- [/data/CLAUDE.md](../data/CLAUDE.md) — Data aggregation, indicator types, `TickAggr`
- [/exchange/CLAUDE.md](../exchange/CLAUDE.md) — ClickHouse adapter, stream types
- [/docs/audits/CLAUDE.md](../docs/audits/CLAUDE.md) — Bar-selection statistical audits

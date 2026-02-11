# Data Crate

**Parent**: [/CLAUDE.md](/CLAUDE.md)

Data aggregation, chart models, indicator types, layout persistence, and configuration. Crate name: `flowsurface-data`.

---

## Quick Reference

| Module           | File                     | Purpose                                         |
| ---------------- | ------------------------ | ----------------------------------------------- |
| Chart basis      | `src/chart.rs`           | `Basis` enum (Time, Tick, RangeBar)             |
| Chart types      | `src/chart/kline.rs`     | `KlineChartKind` (Candles, RangeBar, Footprint) |
| Indicators       | `src/chart/indicator.rs` | `KlineIndicator` enum (6 types)                 |
| Tick aggregation | `src/aggr/ticks.rs`      | `TickAggr`, `RangeBarMicrostructure`            |
| Time aggregation | `src/aggr/time.rs`       | `TimeSeries`                                    |
| Pane layout      | `src/layout/pane.rs`     | `ContentKind` (serialization model)             |
| Timezone labels  | `src/config/timezone.rs` | `format_range_bar_label()` (x-axis)             |

---

## Basis System

The `Basis` enum determines how data is aggregated and rendered on the x-axis:

```rust
pub enum Basis {
    Time(Timeframe),    // Fixed intervals: 1m, 5m, 1h, 4h, 1d, 1w
    Tick(u16),          // N trades per bar
    RangeBar(u32),      // Threshold in dbps (e.g., 250 = 0.25%)
}

pub const RANGE_BAR_THRESHOLDS: [u32; 4] = [250, 500, 750, 1000];
```

| Basis      | Storage      | X-Axis      | Data Source              |
| ---------- | ------------ | ----------- | ------------------------ |
| `Time`     | `TimeSeries` | Continuous  | Exchange WebSocket       |
| `Tick`     | `TickAggr`   | Index-based | Exchange WebSocket       |
| `RangeBar` | `TickAggr`   | Index-based | ClickHouse HTTP (cached) |

**Key difference**: Time-based charts have uniform spacing. Tick and RangeBar charts have non-uniform spacing (index-based, newest rightmost).

---

## Chart Types

```rust
pub enum KlineChartKind {
    Candles,                              // Traditional OHLC candlesticks
    RangeBar,                             // Percentage range bars (fork-specific)
    Footprint { clusters, scaling, studies }, // Price-clustered trade visualization
}
```

The `KlineChartKind` is matched in rendering, scaling, settings, and serialization code. When adding behavior, check all match sites.

---

## Indicator System

### KlineIndicator Enum

```rust
pub enum KlineIndicator {
    Volume,          // Buy/sell volume stacked bars
    OpenInterest,    // Perpetuals only (futures open contracts)
    Delta,           // Buy vol - Sell vol (signed bars)
    TradeCount,      // Trade count histogram (range bars)
    OFI,             // Order Flow Imbalance line (range bars)
    TradeIntensity,  // Trades/sec line (range bars)
}
```

### Market Availability

| Indicator      | Spot | Perps | Source                     |
| -------------- | ---- | ----- | -------------------------- |
| Volume         | Yes  | Yes   | `Kline.volume` (buy, sell) |
| OpenInterest   | —    | Yes   | Exchange stream            |
| Delta          | Yes  | Yes   | `buy_volume - sell_volume` |
| TradeCount     | Yes  | Yes   | `RangeBarMicrostructure`   |
| OFI            | Yes  | Yes   | `RangeBarMicrostructure`   |
| TradeIntensity | Yes  | Yes   | `RangeBarMicrostructure`   |

The last three (TradeCount, OFI, TradeIntensity) only have data for range bar charts — they come from ClickHouse microstructure fields.

### Indicator Trait

```rust
pub trait Indicator: PartialEq + Display + 'static {
    fn for_market(market: MarketKind) -> &'static [Self];
}
```

`FOR_SPOT` (5 types) and `FOR_PERPS` (6 types, +OpenInterest) arrays control which indicators appear in the UI.

---

## Data Aggregation

### TickAggr (Vec-based, oldest-first)

Used for **Tick** and **RangeBar** basis. Bars stored in a `Vec<TickAccumulation>` ordered oldest-first.

```rust
pub struct TickAccumulation {
    pub tick_count: usize,
    pub kline: Kline,
    pub footprint: KlineTrades,
    pub microstructure: Option<RangeBarMicrostructure>,  // Fork-specific
}

pub struct RangeBarMicrostructure {
    pub trade_count: u32,
    pub ofi: f32,
    pub trade_intensity: f32,
}
```

**Extraction methods** on `TickAggr`:

| Method                   | Returns                    | Used By        |
| ------------------------ | -------------------------- | -------------- |
| `volume_data()`          | `BTreeMap<u64, (f32,f32)>` | Volume         |
| `delta_data()`           | `BTreeMap<u64, f32>`       | Delta          |
| `trade_count_data()`     | `BTreeMap<u64, f32>`       | TradeCount     |
| `ofi_data()`             | `BTreeMap<u64, f32>`       | OFI            |
| `trade_intensity_data()` | `BTreeMap<u64, f32>`       | TradeIntensity |

### TimeSeries (time-based)

Used for **Time** basis. Bars keyed by timestamp with fixed intervals.

---

## Pane Serialization

**File**: `src/layout/pane.rs`

Pane state is persisted to JSON for layout save/restore:

```rust
pub enum ContentKind {
    Kline(KlineLayout),
    RangeBarChart(KlineLayout),  // Fork-specific
    Heatmap(HeatmapLayout),
    Footprint(FootprintLayout),
    TimeAndSales(TimeAndSalesLayout),
}
```

---

## X-Axis Labels for Range Bars

**File**: `src/config/timezone.rs` — `format_range_bar_label()`

Two-pass algorithm for non-uniform bar spacing:

1. **Pass 1**: Estimate label density based on visible bar count
2. **Pass 2**: Place labels with minimum spacing, avoiding overlaps

Labels adapt format based on density: time-only for dense, date+time for sparse.

---

## Related

- [/CLAUDE.md](/CLAUDE.md) — Project hub
- [/exchange/CLAUDE.md](/exchange/CLAUDE.md) — Exchange adapters, ClickHouse
- `src/chart/indicator.rs` — Indicator enum source
- `src/aggr/ticks.rs` — TickAggr + RangeBarMicrostructure source

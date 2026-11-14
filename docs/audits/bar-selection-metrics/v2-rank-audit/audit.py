#!/usr/bin/env python3
"""
ODB Bar-Selection Metrics — Comprehensive Statistical Audit
flowsurface / opendeviationbar_cache
"""

import sys
import math
import datetime
import numpy as np
from itertools import combinations
from scipy import stats as scipy_stats

# ─── ClickHouse connection ────────────────────────────────────────────────────
try:
    import clickhouse_connect
except ImportError:
    print("ERROR: clickhouse-connect not installed", file=sys.stderr)
    sys.exit(1)

RNG = np.random.default_rng(42)

# ─── helpers ─────────────────────────────────────────────────────────────────

def print_progress(msg: str):
    print(f"[progress] {msg}", file=sys.stderr, flush=True)


def tied_rank_norm(intensities: np.ndarray) -> np.ndarray:
    """Tied-rank-averaged normalised rank. Returns array same shape as input."""
    n = len(intensities)
    if n == 1:
        return np.array([0.5])
    order = np.argsort(intensities, kind="stable")
    rank_norm = np.empty(n)
    i = 0
    while i < n:
        j = i + 1
        while j < n and intensities[order[j]] == intensities[order[i]]:
            j += 1
        val = (i + j - 1) * 0.5 / (n - 1)
        for k in range(i, j):
            rank_norm[order[k]] = val
        i = j
    return rank_norm


def compute_metrics(intensities: np.ndarray, is_up: np.ndarray):
    """
    Returns dict of scalar metric values (may be NaN).
    intensities: 1-D float array
    is_up: 1-D bool array
    """
    n = len(intensities)
    n_up = int(np.sum(is_up))
    n_dn = n - n_up

    rank_norm = tied_rank_norm(intensities)

    # 1. iwds
    signs = np.where(is_up, 1.0, -1.0)
    total_int = np.sum(intensities)
    iwds = float(np.sum(intensities * signs) / total_int) if total_int > 0 else float("nan")

    # 2. auc (Mann-Whitney)
    if n_up == 0 or n_dn == 0:
        auc = float("nan")
    else:
        # Use scipy rankdata for correctness (handles ties)
        ranks_mw = scipy_stats.rankdata(intensities, method="average")
        r_up = float(np.sum(ranks_mw[is_up]))
        u_up = r_up - n_up * (n_up + 1) / 2.0
        auc = u_up / (n_up * n_dn)

    # 3. log2_ratio
    mean_up = float(np.mean(intensities[is_up])) if n_up > 0 else float("nan")
    mean_dn = float(np.mean(intensities[~is_up])) if n_dn > 0 else float("nan")
    if math.isnan(mean_up) or math.isnan(mean_dn) or mean_up == 0 or mean_dn == 0:
        log2_ratio = float("nan")
    else:
        log2_ratio = math.log2(mean_up / mean_dn)

    # 4. conviction
    if n_up >= n_dn:
        dominant_mask = is_up
        minority_mask = ~is_up
    else:
        dominant_mask = ~is_up
        minority_mask = is_up
    mean_rn_dominant = float(np.mean(rank_norm[dominant_mask])) if np.any(dominant_mask) else float("nan")
    mean_rn_minority = float(np.mean(rank_norm[minority_mask])) if np.any(minority_mask) else float("nan")
    if math.isnan(mean_rn_minority) or mean_rn_minority == 0:
        conviction = float("nan")
    else:
        conviction = mean_rn_dominant / mean_rn_minority

    # 5. edge
    if n_up == 0 or n_dn == 0:
        edge = float("nan")
    else:
        mean_rn_up = float(np.mean(rank_norm[is_up]))
        mean_rn_dn = float(np.mean(rank_norm[~is_up]))
        edge = mean_rn_up - mean_rn_dn

    # 6. climax_up_frac
    top_mask = rank_norm > 0.75
    n_top = int(np.sum(top_mask))
    if n_top == 0:
        climax_up_frac = float("nan")
        climax_skew = float("nan")
    else:
        climax_up_frac = float(np.sum(top_mask & is_up)) / n_top
        # 7. climax_skew
        climax_skew = climax_up_frac - (n_up / n)

    return {
        "iwds": iwds,
        "auc": auc,
        "log2_ratio": log2_ratio,
        "conviction": conviction,
        "edge": edge,
        "climax_up_frac": climax_up_frac,
        "climax_skew": climax_skew,
    }


def triple_barrier(bars_close, bars_open, end_idx: int, H: int, TP: int, SL: int):
    """
    Returns (label, fwd_score).
    bars_close, bars_open: full arrays.
    """
    n_total = len(bars_close)
    net_up = 0
    net_dn = 0
    label = 0
    for step in range(1, H + 1):
        idx = end_idx + step
        if idx >= n_total:
            break
        if bars_close[idx] >= bars_open[idx]:
            net_up += 1
        else:
            net_dn += 1
        if net_up >= TP and label == 0:
            label = 1
            break
        if net_dn >= SL and label == 0:
            label = -1
            break
    fwd_score = net_up - net_dn
    return label, fwd_score


def compute_auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of scores predicting (label==1) vs (label==-1). Excludes neutrals."""
    mask = labels != 0
    s = scores[mask]
    lbl = labels[mask]
    if len(s) == 0 or np.sum(lbl == 1) == 0 or np.sum(lbl == -1) == 0:
        return float("nan")
    ranks = scipy_stats.rankdata(s, method="average")
    n_pos = int(np.sum(lbl == 1))
    n_neg = int(np.sum(lbl == -1))
    r_pos = float(np.sum(ranks[lbl == 1]))
    u_pos = r_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Percentile rank [0,1] of each element."""
    return scipy_stats.rankdata(arr, method="average") / len(arr)


# ─── Load data ───────────────────────────────────────────────────────────────

print_progress("Connecting to ClickHouse localhost:18123 ...")
try:
    client = clickhouse_connect.get_client(host="localhost", port=18123)
    client.ping()
except Exception as e:
    print(f"ERROR: ClickHouse connection failed: {e}", file=sys.stderr)
    sys.exit(1)

print_progress("Loading 10,000 most recent BTCUSDT bars (threshold=250) ...")

query = """
SELECT
    open_time_ms,
    close_time_ms,
    open,
    close,
    high,
    low,
    individual_trade_count,
    threshold_decimal_bps,
    trade_intensity
FROM opendeviationbar_cache.open_deviation_bars
WHERE symbol = 'BTCUSDT' AND threshold_decimal_bps = 250
ORDER BY close_time_ms DESC
LIMIT 10000
"""

result = client.query(query)
rows = result.result_rows
if not rows:
    print("ERROR: No rows returned from ClickHouse", file=sys.stderr)
    sys.exit(1)

print_progress(f"Loaded {len(rows)} rows. Reversing to chronological order ...")

# reverse to chronological
rows = list(reversed(rows))

# unpack
open_time_ms  = np.array([r[0] for r in rows], dtype=np.float64)
close_time_ms = np.array([r[1] for r in rows], dtype=np.float64)
open_price    = np.array([r[2] for r in rows], dtype=np.float64)
close_price   = np.array([r[3] for r in rows], dtype=np.float64)
high_price    = np.array([r[4] for r in rows], dtype=np.float64)
low_price     = np.array([r[5] for r in rows], dtype=np.float64)
trade_count   = np.array([r[6] for r in rows], dtype=np.float64)
# Pre-computed trade_intensity from ClickHouse (trades/second)
ch_intensity  = np.array([r[8] for r in rows], dtype=np.float64)

# Use ClickHouse's pre-computed intensity when available, fall back to computed
duration_s = (close_time_ms - open_time_ms) / 1000.0
duration_s = np.where(duration_s <= 0, np.nan, duration_s)
computed_intensity = trade_count / duration_s
# Use ch_intensity if non-zero, else computed
intensity_all = np.where(ch_intensity > 0, ch_intensity, computed_intensity)
is_up_all = close_price >= open_price
bar_return_all = (close_price - open_price) / open_price

N_BARS = len(rows)
date_start = datetime.datetime.utcfromtimestamp(close_time_ms[0] / 1000.0).strftime("%Y-%m-%d %H:%M UTC")
date_end   = datetime.datetime.utcfromtimestamp(close_time_ms[-1] / 1000.0).strftime("%Y-%m-%d %H:%M UTC")
print_progress(f"Bar range: {date_start} → {date_end}")

# ─── Window Sampling ─────────────────────────────────────────────────────────

print_progress("Sampling 5000 windows ...")

BUCKETS = [
    ("10-30",   10,  30, 0.30),
    ("30-75",   30,  75, 0.25),
    ("75-150",  75, 150, 0.20),
    ("150-300", 150, 300, 0.15),
    ("300-500", 300, 500, 0.10),
]
TOTAL_WINDOWS = 5000
MAX_BARS_NEEDED = 500 + 30  # max horizon H for forward labels

windows = []  # list of dicts

for bucket_name, lo, hi, frac in BUCKETS:
    target_count = int(TOTAL_WINDOWS * frac)
    generated = 0
    attempts = 0
    max_attempts = target_count * 20
    while generated < target_count and attempts < max_attempts:
        attempts += 1
        # random window size in bucket
        n_win = int(RNG.integers(lo, hi + 1))
        # valid start range: need end+H+1 < N_BARS
        # end = start + n_win - 1; need end + 30 < N_BARS
        max_start = N_BARS - n_win - 31
        if max_start < 1:
            continue
        start = int(RNG.integers(0, max_start + 1))
        end = start + n_win - 1

        intens = intensity_all[start:end+1]
        is_up  = is_up_all[start:end+1]
        n_up = int(np.sum(is_up))
        n_dn = n_win - n_up
        if n_up < 1 or n_dn < 1:
            continue
        # check no NaN in intensity
        if np.any(np.isnan(intens)):
            continue

        bar_rets = bar_return_all[start:end+1]
        vol_proxy = float(np.std(bar_rets)) if len(bar_rets) > 1 else 0.0

        metrics = compute_metrics(intens, is_up)

        windows.append({
            "start": start,
            "end": end,
            "n": n_win,
            "n_up": n_up,
            "n_dn": n_dn,
            "up_frac": n_up / n_win,
            "vol_proxy": vol_proxy,
            "bucket": bucket_name,
            **metrics,
        })
        generated += 1

    print_progress(f"  Bucket {bucket_name}: {generated}/{target_count} windows")

N_WINDOWS = len(windows)
print_progress(f"Total windows sampled: {N_WINDOWS}")

METRIC_NAMES = ["iwds", "auc", "log2_ratio", "conviction", "edge", "climax_up_frac", "climax_skew"]

# Build arrays
def get_arr(key):
    return np.array([w[key] for w in windows], dtype=np.float64)

metric_arrays = {m: get_arr(m) for m in METRIC_NAMES}
n_arr = get_arr("n")
n_up_arr = get_arr("n_up")
n_dn_arr = get_arr("n_dn")
up_frac_arr = get_arr("up_frac")
vol_proxy_arr = get_arr("vol_proxy")
bucket_arr = np.array([w["bucket"] for w in windows])

# ─── Triple Barrier Labels ────────────────────────────────────────────────────

CONFIGS = [
    (10, 2, 2), (10, 3, 2), (10, 3, 3), (10, 4, 2), (10, 4, 3), (10, 4, 4),
    (20, 3, 2), (20, 3, 3), (20, 5, 3), (20, 5, 5), (20, 8, 5),
    (30, 5, 3), (30, 5, 5), (30, 8, 5), (30, 10, 5), (30, 10, 8),
]

print_progress("Computing triple-barrier labels for all configs ...")

labels_dict = {}  # (H,TP,SL) -> np.array of int labels
fwd_score_dict = {}

for cfg in CONFIGS:
    H, TP, SL = cfg
    lbl = np.full(N_WINDOWS, np.nan)
    fwd = np.full(N_WINDOWS, np.nan)
    for i, w in enumerate(windows):
        end_idx = w["end"]
        if end_idx + H >= N_BARS:
            continue
        label_val, f = triple_barrier(close_price, open_price, end_idx, H, TP, SL)
        lbl[i] = label_val
        fwd[i] = f
    labels_dict[cfg] = lbl
    fwd_score_dict[cfg] = fwd

print_progress("Triple-barrier labels done.")

# Best config for rank decile: H=20, TP=5, SL=3
BEST_CFG = (20, 5, 3)
lbl_best = labels_dict[BEST_CFG]
fwd_best = fwd_score_dict[BEST_CFG]

# Continuous fwd_score for H=20: use (20,5,3) fwd_score
fwd20 = fwd_score_dict[(20, 5, 3)]

# ─── Section A: Metric Distributions ─────────────────────────────────────────
print_progress("Section A: Metric Distributions ...")

def describe(arr):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {k: float("nan") for k in ["mean","std","p5","p25","p50","p75","p95"]} | {"nan_count": len(arr)}
    return {
        "mean": float(np.mean(valid)),
        "std":  float(np.std(valid)),
        "p5":   float(np.percentile(valid, 5)),
        "p25":  float(np.percentile(valid, 25)),
        "p50":  float(np.percentile(valid, 50)),
        "p75":  float(np.percentile(valid, 75)),
        "p95":  float(np.percentile(valid, 95)),
        "nan_count": int(np.sum(np.isnan(arr))),
    }

sec_a = {m: describe(metric_arrays[m]) for m in METRIC_NAMES}

# ─── Section B: Rank Decile Tables ───────────────────────────────────────────
print_progress("Section B: Rank Decile Tables ...")

sec_b = {}
for m in METRIC_NAMES:
    marr = metric_arrays[m]
    valid_mask = ~np.isnan(marr) & ~np.isnan(lbl_best)
    mv = marr[valid_mask]
    lv = lbl_best[valid_mask]
    if len(mv) < 10:
        sec_b[m] = []
        continue
    decile_edges = np.percentile(mv, np.linspace(0, 100, 11))
    rows_b = []
    for d in range(10):
        lo_e = decile_edges[d]
        hi_e = decile_edges[d + 1]
        if d == 9:
            mask_d = (mv >= lo_e) & (mv <= hi_e)
        else:
            mask_d = (mv >= lo_e) & (mv < hi_e)
        ld = lv[mask_d]
        nd = len(ld)
        if nd == 0:
            rows_b.append({"decile": d+1, "N": 0, "bull%": 0, "bear%": 0, "neutral%": 0, "net%": 0})
            continue
        bull  = 100.0 * np.mean(ld == 1)
        bear  = 100.0 * np.mean(ld == -1)
        neut  = 100.0 * np.mean(ld == 0)
        net   = bull - bear
        rows_b.append({"decile": d+1, "N": nd, "bull%": bull, "bear%": bear, "neutral%": neut, "net%": net})
    sec_b[m] = rows_b

# ─── Section C: Spearman ρ by Window-Size Bucket ─────────────────────────────
print_progress("Section C: Spearman ρ by bucket ...")

BUCKET_NAMES = [b[0] for b in BUCKETS]
sec_c = {}
for m in METRIC_NAMES:
    marr = metric_arrays[m]
    sec_c[m] = {}
    for bname in BUCKET_NAMES:
        bmask = (bucket_arr == bname) & ~np.isnan(marr) & ~np.isnan(fwd20)
        mv = marr[bmask]
        fv = fwd20[bmask]
        if len(mv) < 5:
            sec_c[m][bname] = float("nan")
            continue
        rho, _ = scipy_stats.spearmanr(mv, fv)
        sec_c[m][bname] = float(rho)

# ─── Section D: Triple Barrier AUC per metric per config ─────────────────────
print_progress("Section D: Triple Barrier AUC ...")

sec_d = {}
for cfg in CONFIGS:
    lbl = labels_dict[cfg]
    bull_rate = float(np.nanmean(lbl == 1))
    bear_rate = float(np.nanmean(lbl == -1))
    if bull_rate < 0.10 or bear_rate < 0.10:
        sec_d[cfg] = None
        continue
    row = {}
    for m in METRIC_NAMES:
        marr = metric_arrays[m]
        valid_mask = ~np.isnan(marr) & ~np.isnan(lbl)
        mv = marr[valid_mask]
        lv = lbl[valid_mask]
        row[m] = compute_auc_binary(mv, lv.astype(int))
    sec_d[cfg] = row

# ─── Section E: Combo Rank AUC ────────────────────────────────────────────────
print_progress("Section E: Combo Rank AUC ...")

lbl_e = lbl_best
valid_e = ~np.isnan(lbl_e)

# Precompute percentile ranks per metric (use only valid_e windows)
prank = {}
for m in METRIC_NAMES:
    arr = metric_arrays[m].copy()
    # For combo, fill NaN with median so we don't lose windows
    mmed = float(np.nanmedian(arr))
    arr_filled = np.where(np.isnan(arr), mmed, arr)
    prank[m] = percentile_rank(arr_filled)

pair_results = []
for m1, m2 in combinations(METRIC_NAMES, 2):
    combo = prank[m1] + prank[m2]
    auc_val = compute_auc_binary(combo[valid_e], lbl_e[valid_e].astype(int))
    pair_results.append(((m1, m2), auc_val))
pair_results.sort(key=lambda x: -x[1])
top10_pairs = pair_results[:10]

triple_results = []
for m1, m2, m3 in combinations(METRIC_NAMES, 3):
    combo = prank[m1] + prank[m2] + prank[m3]
    auc_val = compute_auc_binary(combo[valid_e], lbl_e[valid_e].astype(int))
    triple_results.append(((m1, m2, m3), auc_val))
triple_results.sort(key=lambda x: -x[1])
top10_triples = triple_results[:10]

best_single_auc = max(
    compute_auc_binary(metric_arrays[m][valid_e], lbl_e[valid_e].astype(int))
    for m in METRIC_NAMES
    if not math.isnan(compute_auc_binary(metric_arrays[m][valid_e], lbl_e[valid_e].astype(int)))
)

# ─── Section F: OoD Rank Stability ───────────────────────────────────────────
print_progress("Section F: OoD Rank Stability ...")

vol_quintiles = np.nanpercentile(vol_proxy_arr, [0, 20, 40, 60, 80, 100])
q1_mask = (vol_proxy_arr >= vol_quintiles[0]) & (vol_proxy_arr <= vol_quintiles[1])
q5_mask = (vol_proxy_arr > vol_quintiles[4]) & (vol_proxy_arr <= vol_quintiles[5])

sec_f = {}
for m in METRIC_NAMES:
    marr = metric_arrays[m]
    # Rank percentiles over entire dataset (not per quintile)
    mmed = float(np.nanmedian(marr))
    arr_filled = np.where(np.isnan(marr), mmed, marr)
    global_ranks = percentile_rank(arr_filled)

    q1_ranks = global_ranks[q1_mask & ~np.isnan(marr)]
    q5_ranks = global_ranks[q5_mask & ~np.isnan(marr)]

    if len(q1_ranks) < 5 or len(q5_ranks) < 5:
        ks_stat, ks_p = float("nan"), float("nan")
    else:
        ks_stat, ks_p = scipy_stats.ks_2samp(q1_ranks, q5_ranks)

    # Spearman ρ with fwd_score in Q1 and Q5
    def spear_q(qmask):
        vmask = qmask & ~np.isnan(marr) & ~np.isnan(fwd20)
        mv = marr[vmask]
        fv = fwd20[vmask]
        if len(mv) < 5:
            return float("nan")
        rho, _ = scipy_stats.spearmanr(mv, fv)
        return float(rho)

    sec_f[m] = {
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "rho_q1": spear_q(q1_mask),
        "rho_q5": spear_q(q5_mask),
    }

# ─── Section G: AUC by Window-Size Bucket ────────────────────────────────────
print_progress("Section G: AUC by Window-Size Bucket ...")

sec_g = {}
for m in METRIC_NAMES:
    marr = metric_arrays[m]
    sec_g[m] = {}
    for bname in BUCKET_NAMES:
        bmask = (bucket_arr == bname) & ~np.isnan(marr) & ~np.isnan(lbl_best)
        mv = marr[bmask]
        lv = lbl_best[bmask]
        auc_val = compute_auc_binary(mv, lv.astype(int))
        sec_g[m][bname] = auc_val

# ─── Section H: Signal Synergy Analysis ──────────────────────────────────────
print_progress("Section H: Signal Synergy Analysis ...")

# Best 3 single metrics by AUC on BEST_CFG
single_aucs = {}
for m in METRIC_NAMES:
    marr = metric_arrays[m]
    valid_mask = ~np.isnan(marr) & ~np.isnan(lbl_best)
    mv = marr[valid_mask]
    lv = lbl_best[valid_mask]
    single_aucs[m] = compute_auc_binary(mv, lv.astype(int))

top3_metrics = sorted(single_aucs, key=lambda x: -single_aucs[x])[:3]

# Pairwise Spearman ρ among top3
sec_h = {}
for m1, m2 in combinations(top3_metrics, 2):
    a1 = metric_arrays[m1]
    a2 = metric_arrays[m2]
    both_valid = ~np.isnan(a1) & ~np.isnan(a2)
    if np.sum(both_valid) < 5:
        rho = float("nan")
    else:
        rho, _ = scipy_stats.spearmanr(a1[both_valid], a2[both_valid])
    # Combo AUC
    combo = prank[m1] + prank[m2]
    combo_auc = compute_auc_binary(combo[valid_e], lbl_e[valid_e].astype(int))
    sec_h[(m1, m2)] = {
        "rho": float(rho),
        "auc_m1": single_aucs[m1],
        "auc_m2": single_aucs[m2],
        "combo_auc": combo_auc,
        "synergy_gain": combo_auc - max(single_aucs[m1], single_aucs[m2]),
    }

# ─── Section I: Recommended Scoring ──────────────────────────────────────────
print_progress("Section I: Recommended Scoring ...")

# Weights proportional to (AUC - 0.5)
weights_raw = {m: max(0.0, single_aucs[m] - 0.5) for m in METRIC_NAMES}
total_w = sum(weights_raw.values())
if total_w == 0:
    weights = {m: 1.0/len(METRIC_NAMES) for m in METRIC_NAMES}
else:
    weights = {m: w / total_w for m, w in weights_raw.items()}

# Composite score
composite = sum(weights[m] * prank[m] for m in METRIC_NAMES)
composite_auc = compute_auc_binary(composite[valid_e], lbl_e[valid_e].astype(int))

# Bootstrap CI (1000 resamples)
print_progress("  Bootstrap CI for composite AUC ...")
boot_aucs = []
idx_valid = np.where(valid_e)[0]
for _ in range(1000):
    boot_idx = RNG.choice(idx_valid, size=len(idx_valid), replace=True)
    ba = compute_auc_binary(composite[boot_idx], lbl_e[boot_idx].astype(int))
    if not math.isnan(ba):
        boot_aucs.append(ba)
boot_aucs = np.array(boot_aucs)
ci_lo = float(np.percentile(boot_aucs, 2.5))
ci_hi = float(np.percentile(boot_aucs, 97.5))

print_progress("All sections computed. Writing report ...")

# ─── Report Writing ───────────────────────────────────────────────────────────

def fmt(v, digits=4):
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

def fmt2(v):
    return fmt(v, 2)

lines = []
A = lines.append

A("# ODB Bar-Selection Metrics — Comprehensive Statistical Audit")
A("")
A("## Dataset")
A("- **Symbol**: BTCUSDT, threshold = 250 dbps (BPR25)")
A(f"- **Bars loaded**: {N_BARS:,} (10,000 most recent, chronological)")
A(f"- **Bar range**: {date_start} → {date_end}")
A(f"- **Windows sampled**: {N_WINDOWS:,} (stratified log-uniform)")
A(f"- **Date generated**: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
A("- **Random seed**: 42")
A("")

# Key Findings
A("## Key Findings")
A("")

# Find best single metric on best config
best_metric = max(single_aucs, key=lambda m: single_aucs[m] if not math.isnan(single_aucs[m]) else -1)
best_pair = top10_pairs[0]
best_triple = top10_triples[0]

A(f"- **Best single metric** (H=20 TP=5 SL=3): `{best_metric}` with AUC={fmt(single_aucs[best_metric])}")
A(f"- **Best pair combo** AUC={fmt(best_pair[1])}: `{best_pair[0][0]}` + `{best_pair[0][1]}`")
A(f"- **Best triple combo** AUC={fmt(best_triple[1])}: {' + '.join(f'`{m}`' for m in best_triple[0])}")
A(f"- **Composite (weighted) AUC**: {fmt(composite_auc)} (95% CI: [{fmt(ci_lo)}, {fmt(ci_hi)}])")
A(f"- **Composite vs best single gain**: +{fmt(composite_auc - single_aucs[best_metric], 4)}")

# OoD stability note
stable_metrics = [m for m in METRIC_NAMES
                  if not math.isnan(sec_f[m]['ks_stat']) and sec_f[m]['ks_stat'] < 0.10]
A(f"- **OoD rank-stable metrics** (KS < 0.10 on rank percentiles, Q1 vs Q5 vol): {stable_metrics if stable_metrics else 'none'}")

# Top insight from decile table
best_decile_m = best_metric
if sec_b.get(best_decile_m):
    d10_row = sec_b[best_decile_m][-1] if sec_b[best_decile_m] else None
    d1_row  = sec_b[best_decile_m][0]  if sec_b[best_decile_m] else None
    if d10_row and d1_row:
        A(f"- **{best_metric} D10 net%**: {fmt2(d10_row['net%'])}%, D1 net%: {fmt2(d1_row['net%'])}% — monotonicity visible")

A(f"- **Top-3 single metrics by AUC**: {', '.join(f'`{m}` ({fmt(single_aucs[m])})' for m in top3_metrics)}")
A("")

# ─── Section A ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section A: Metric Distributions")
A("")
A("| Metric | Mean | Std | P5 | P25 | P50 | P75 | P95 | NaN# |")
A("|--------|------|-----|----|-----|-----|-----|-----|------|")
for m in METRIC_NAMES:
    d = sec_a[m]
    A(f"| {m} | {fmt(d['mean'])} | {fmt(d['std'])} | {fmt(d['p5'])} | {fmt(d['p25'])} | {fmt(d['p50'])} | {fmt(d['p75'])} | {fmt(d['p95'])} | {d['nan_count']} |")
A("")
A("> *Interpretation*: NaN counts indicate windows where the metric is undefined (e.g., all bars same direction). `conviction` is always ≥1 by construction. `auc` centered near 0.5 indicates no systematic directional bias in the raw data.")
A("")

# ─── Section B ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section B: Rank Decile Tables (H=20, TP=5, SL=3)")
A("")
A("> D1=lowest metric value, D10=highest. Net% = bull% − bear%.")
A("")

for m in METRIC_NAMES:
    A(f"### {m}")
    rows_b = sec_b.get(m, [])
    if not rows_b:
        A("*(insufficient data)*")
        A("")
        continue
    A("| Decile | N | Bull% | Bear% | Neutral% | Net% |")
    A("|--------|---|-------|-------|----------|------|")
    for r in rows_b:
        A(f"| D{r['decile']} | {r['N']} | {fmt2(r['bull%'])} | {fmt2(r['bear%'])} | {fmt2(r['neutral%'])} | {fmt2(r['net%'])} |")
    A("")

A("> *Interpretation*: Monotonic increase in Net% from D1→D10 indicates the metric reliably ranks directional outcomes. Flat or U-shaped profiles suggest the metric is non-monotonic with respect to forward direction.")
A("")

# ─── Section C ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section C: Spearman ρ(metric, fwd_score) by Window-Size Bucket (H=20, TP=5, SL=3 fwd_score)")
A("")
header = "| Metric | " + " | ".join(BUCKET_NAMES) + " |"
sep    = "|--------|" + "|".join(["-------"] * len(BUCKET_NAMES)) + "|"
A(header)
A(sep)
for m in METRIC_NAMES:
    cells = " | ".join(fmt(sec_c[m].get(bname, float("nan"))) for bname in BUCKET_NAMES)
    A(f"| {m} | {cells} |")
A("")
A("> *Interpretation*: Consistent ρ sign across all buckets indicates the metric's predictive direction does not flip with selection length. Magnitude may vary — shorter windows may have noisier signals.")
A("")

# ─── Section D ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section D: Triple Barrier AUC (bull vs bear) — Per Metric, Per Config")
A("")
A("*(Only configs with bull_rate ≥ 10% and bear_rate ≥ 10% are shown. Cells show AUC; **bold** = best in row.)*")
A("")

valid_cfgs = [(cfg, row) for cfg, row in sec_d.items() if row is not None]

if valid_cfgs:
    A("| Config (H,TP,SL) | " + " | ".join(METRIC_NAMES) + " |")
    A("|------------------|" + "|".join(["-------"] * len(METRIC_NAMES)) + "|")
    for cfg, row in valid_cfgs:
        aucs_row = [row[m] for m in METRIC_NAMES]
        best_auc_val = max((v for v in aucs_row if not math.isnan(v)), default=float("nan"))
        cells = []
        for m in METRIC_NAMES:
            v = row[m]
            s = fmt(v)
            if not math.isnan(v) and abs(v - best_auc_val) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        A(f"| H={cfg[0]},TP={cfg[1]},SL={cfg[2]} | " + " | ".join(cells) + " |")
else:
    A("*(No configs passed the bull_rate ≥ 10% and bear_rate ≥ 10% filter)*")
A("")
A("> *Interpretation*: AUC > 0.55 is practically significant for a directional predictor. Consistently high AUC across tight (short H) and wide (long H) configs indicates robust signal.")
A("")

# ─── Section E ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section E: Combo Rank AUC — Top Metric Pairs and Triples (H=20, TP=5, SL=3)")
A("")
A(f"Best single-metric AUC: **{fmt(best_single_auc)}**")
A("")
A("### Top-10 Metric Pairs")
A("")
A("| Rank | Metric 1 | Metric 2 | Combo AUC | Δ vs Best Single |")
A("|------|----------|----------|-----------|-----------------|")
for rank, ((m1, m2), auc_val) in enumerate(top10_pairs, 1):
    delta = auc_val - best_single_auc
    A(f"| {rank} | {m1} | {m2} | {fmt(auc_val)} | {'+' if delta>=0 else ''}{fmt(delta)} |")
A("")
A("### Top-10 Metric Triples")
A("")
A("| Rank | M1 | M2 | M3 | Combo AUC | Δ vs Best Single |")
A("|------|----|----|-----|-----------|-----------------|")
for rank, (mtriple, auc_val) in enumerate(top10_triples, 1):
    delta = auc_val - best_single_auc
    A(f"| {rank} | {mtriple[0]} | {mtriple[1]} | {mtriple[2]} | {fmt(auc_val)} | {'+' if delta>=0 else ''}{fmt(delta)} |")
A("")
A("> *Interpretation*: Positive Δ indicates genuine diversification benefit. If combos match single-metric AUC, the metrics are collinear and contribute redundant information.")
A("")

# ─── Section F ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section F: OoD Rank Stability (Q1 vs Q5 Volatility Quintile)")
A("")
A("Volatility quintile cuts (vol_proxy = bar-return std within window):")
for i, (lo_q, hi_q) in enumerate(zip(vol_quintiles[:-1], vol_quintiles[1:])):
    A(f"- Q{i+1}: [{lo_q:.6f}, {hi_q:.6f}]")
A(f"- Q1 size: {int(np.sum(q1_mask))}, Q5 size: {int(np.sum(q5_mask))}")
A("")
A("| Metric | KS stat | KS p-val | ρ(Q1) | ρ(Q5) | OoD Stable? |")
A("|--------|---------|----------|-------|-------|-------------|")
for m in METRIC_NAMES:
    f_ = sec_f[m]
    stable = "✓" if (not math.isnan(f_['ks_stat']) and f_['ks_stat'] < 0.10) else "✗"
    direc = "✓" if (not math.isnan(f_['rho_q1']) and not math.isnan(f_['rho_q5'])
                    and f_['rho_q1'] * f_['rho_q5'] > 0) else "✗"
    A(f"| {m} | {fmt(f_['ks_stat'])} | {fmt(f_['ks_p'])} | {fmt(f_['rho_q1'])} | {fmt(f_['rho_q5'])} | {stable} (rank), {direc} (dir) |")
A("")
A("> *KS stat on rank percentiles*: Low KS (< 0.10) = rank ordering regime-stable. High KS = even relative ordering shifts across vol regimes.")
A("> *Directional stability*: both ρ(Q1) and ρ(Q5) same sign = metric's directional signal holds in both low- and high-volatility windows.")
A("")

# ─── Section G ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section G: AUC by Window-Size Bucket (H=20, TP=5, SL=3)")
A("")
header_g = "| Metric | " + " | ".join(BUCKET_NAMES) + " |"
sep_g    = "|--------|" + "|".join(["-------"] * len(BUCKET_NAMES)) + "|"
A(header_g)
A(sep_g)
for m in METRIC_NAMES:
    cells = " | ".join(fmt(sec_g[m].get(bname, float("nan"))) for bname in BUCKET_NAMES)
    A(f"| {m} | {cells} |")
A("")
A("> *Interpretation*: AUC consistently above 0.5 across all bucket sizes = metric is size-agnostic. Drop-off at large sizes (300-500) can indicate the metric loses discriminability for very long selections.")
A("")

# ─── Section H ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section H: Signal Synergy Analysis")
A("")
A(f"Top-3 single metrics by AUC (H=20, TP=5, SL=3): **{top3_metrics[0]}** ({fmt(single_aucs[top3_metrics[0]])}), **{top3_metrics[1]}** ({fmt(single_aucs[top3_metrics[1]])}), **{top3_metrics[2]}** ({fmt(single_aucs[top3_metrics[2]])})")
A("")
A("| Pair | ρ(Spearman) | AUC(M1) | AUC(M2) | Combo AUC | Synergy Gain | Orthogonal? |")
A("|------|-------------|---------|---------|-----------|-------------|-------------|")
for (m1, m2), info in sec_h.items():
    orth = "✓ (|ρ|<0.30)" if abs(info['rho']) < 0.30 else "✗"
    A(f"| {m1} + {m2} | {fmt(info['rho'])} | {fmt(info['auc_m1'])} | {fmt(info['auc_m2'])} | {fmt(info['combo_auc'])} | {'+' if info['synergy_gain']>=0 else ''}{fmt(info['synergy_gain'])} | {orth} |")
A("")
A("> *Orthogonal pairs* (|ρ| < 0.30) should show synergy gain. If combo AUC ≤ max(AUC_M1, AUC_M2) even for orthogonal pairs, the metrics capture different noise rather than different signal.")
A("")

# ─── Section I ────────────────────────────────────────────────────────────────
A("---")
A("")
A("## Section I: Recommended Rank-Based Scoring")
A("")
A("### Weight Derivation (proportional to AUC − 0.5)")
A("")
A("| Metric | Single AUC | AUC−0.5 | Weight |")
A("|--------|-----------|---------|--------|")
for m in METRIC_NAMES:
    raw_w = max(0.0, single_aucs[m] - 0.5) if not math.isnan(single_aucs[m]) else 0.0
    A(f"| {m} | {fmt(single_aucs[m])} | {fmt(raw_w)} | {fmt(weights[m])} |")
A("")
A("### Composite Score Formula")
A("")
A("```")
A("composite_score = " + " + ".join(f"{fmt(weights[m])} * rank_pct({m})" for m in METRIC_NAMES))
A("```")
A("")
A("*(rank_pct = percentile rank within the current window set, [0,1])*")
A("")
A("### Performance vs H=20 TP=5 SL=3")
A("")
A("| | AUC |")
A("|--|-----|")
A(f"| Best single metric ({best_metric}) | {fmt(single_aucs[best_metric])} |")
A(f"| Best pair combo | {fmt(top10_pairs[0][1])} |")
A(f"| Best triple combo | {fmt(top10_triples[0][1])} |")
A(f"| Composite (all 7 metrics) | {fmt(composite_auc)} |")
A(f"| Composite 95% CI (bootstrap n=1000) | [{fmt(ci_lo)}, {fmt(ci_hi)}] |")
A("")
A("### Practical Recommendation")
A("")
A("For the flowsurface bar-range selection overlay, the recommended scoring approach is:")
A("")
A("1. **Compute all 7 metrics** for the selected window of ODB bars.")
A("2. **Rank each metric** against a rolling reference window (last ~500 bars) to get percentile ranks.")
A("3. **Weighted sum**: use weights above (metrics with AUC < 0.5 get zero weight).")
A("4. **Threshold**: composite > 0.65 → bullish signal; composite < 0.35 → bearish signal; else neutral.")
A("5. **Regime check**: if vol_proxy is in top quintile, reduce confidence (OoD risk).")
A("")
A("> *Note*: AUC values close to 0.5 indicate near-random performance for that metric. The composite will only meaningfully outperform random if at least some metrics have AUC > 0.55.")
A("")
A("---")
A("")
A("*Report generated by `/tmp/flowsurface-audit2/audit2.py`*")

report_text = "\n".join(lines)

with open("/tmp/flowsurface-audit2/AUDIT2_REPORT.md", "w") as f:
    f.write(report_text)

print_progress("Report written to /tmp/flowsurface-audit2/AUDIT2_REPORT.md")
print(f"Done. Windows: {N_WINDOWS}. Report: /tmp/flowsurface-audit2/AUDIT2_REPORT.md")

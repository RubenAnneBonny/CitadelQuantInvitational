"""
Lookback Window Optimizer
==========================
Finds the optimal number of ticks (N) to use when fitting spread
parameters and calibrating thresholds in live_pair_trader.py.

Method — walk-forward simulation:
  For each candidate N in WINDOW_CANDIDATES:
    For each starting tick t (stepping by STEP):
      1. Take the N ticks ending at t  →  fit OLS (coef, intercept, SD)
      2. Run _calibrate_thresholds on those same N ticks
      3. Simulate pair trading on the next TEST_WINDOW ticks
      4. Record the Sharpe of that out-of-sample window
    Score for N  =  mean Sharpe across all starting points

Prints a ranked table and a bar chart.

Usage
-----
    python optimize_lookback.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from daily_pair_regression import run as get_daily_pairs

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE        = os.path.join(os.path.dirname(__file__), "../Competition/Alpha Testing V2/data.csv")
PAIR_RANK        = 1          # which pair from daily_pair_regression to test
CAPITAL          = 1_000_000
TRADE_FRACTION   = 0.25

WINDOW_CANDIDATES = list(range(10, 201, 10))   # 10, 20, 30 … 200 ticks
TEST_WINDOW       = 100    # ticks to simulate forward after each fit
STEP              = 20     # advance starting point by this many ticks each walk
GRID_STEPS        = 20     # resolution of the threshold grid search
# ─────────────────────────────────────────────────────────────────────────────


def _calibrate_thresholds(spread_arr, sd, grid_steps=GRID_STEPS):
    """Same logic as in live_pair_trader.py."""
    if sd == 0:
        return 0.5, 0.1

    mults = np.linspace(0.1, 4.0, grid_steps)
    best_sharpe = -np.inf
    best_entry  = sd * 1.0
    best_exit   = sd * 0.3

    for em in mults:
        for xm in mults:
            if xm >= em:
                continue
            entry = em * sd
            exit_ = xm * sd

            in_pos    = 0
            entry_val = 0.0
            trades    = []

            for s in spread_arr:
                if in_pos == 0:
                    if s >= entry:
                        in_pos = -1; entry_val = s
                    elif s <= -entry:
                        in_pos = +1; entry_val = s
                elif in_pos == -1 and s <= exit_:
                    trades.append(entry_val - s)
                    in_pos = 0
                elif in_pos == +1 and s >= -exit_:
                    trades.append(s - entry_val)
                    in_pos = 0

            if len(trades) < 2:
                continue
            t  = np.array(trades)
            sh = t.mean() / (t.std() + 1e-9)
            if sh > best_sharpe:
                best_sharpe = sh
                best_entry  = entry
                best_exit   = exit_

    return best_entry, best_exit


def _simulate(prices1, prices2, coef, intercept, entry_thresh, exit_thresh):
    """
    Simulate two-sided pair trading on arrays prices1 / prices2.
    Returns tick-level Sharpe of the MTM P&L, or np.nan if no variance.
    """
    notional = CAPITAL * TRADE_FRACTION
    tot1 = 0.0
    tot2 = 0.0
    in_pos = 0   # 0=flat, +1=long spread, -1=short spread
    pnl_history = []

    for p1, p2 in zip(prices1, prices2):
        spread = p2 - (coef * p1 + intercept)
        mtm    = tot1 * p1 + tot2 * p2
        pnl_history.append(mtm)

        qty2 = int(notional // p2)
        qty1 = int(notional // p1)

        if in_pos == 0:
            if spread >= entry_thresh:
                tot2 = -qty2; tot1 = qty1; in_pos = -1
            elif spread <= -entry_thresh:
                tot2 = qty2; tot1 = -qty1; in_pos = 1
        elif in_pos == -1 and spread <= exit_thresh:
            tot2 = 0.0; tot1 = 0.0; in_pos = 0
        elif in_pos == 1 and spread >= -exit_thresh:
            tot2 = 0.0; tot1 = 0.0; in_pos = 0

    if len(pnl_history) < 2:
        return np.nan
    deltas = np.diff(pnl_history)
    sigma  = deltas.std()
    if sigma == 0:
        return np.nan
    return float(deltas.mean() / sigma)


def evaluate_window(prices1, prices2, N):
    """
    Walk-forward evaluation for a single window size N.
    Returns list of out-of-sample Sharpe values.
    """
    total = len(prices1)
    sharpes = []

    for t in range(N, total - TEST_WINDOW, STEP):
        # ── Fit on last N ticks ending at t ───────────────────────────────────
        h1 = prices1[t - N : t]
        h2 = prices2[t - N : t]

        model     = LinearRegression(fit_intercept=True).fit(h1.reshape(-1, 1), h2)
        coef      = float(model.coef_[0])
        intercept = float(model.intercept_)
        residuals = h2 - (coef * h1 + intercept)
        sd        = float(residuals.std())

        entry_thresh, exit_thresh = _calibrate_thresholds(residuals, sd)

        # ── Simulate on the next TEST_WINDOW ticks ────────────────────────────
        f1 = prices1[t : t + TEST_WINDOW]
        f2 = prices2[t : t + TEST_WINDOW]
        sh = _simulate(f1, f2, coef, intercept, entry_thresh, exit_thresh)
        if not np.isnan(sh):
            sharpes.append(sh)

    return sharpes


def run():
    df = pd.read_csv(DATA_FILE)

    print(f"Running daily pair regression — picking rank #{PAIR_RANK} pair...")
    _, _, _, pairs = get_daily_pairs()
    if PAIR_RANK > len(pairs):
        raise ValueError(f"PAIR_RANK={PAIR_RANK} but only {len(pairs)} pairs found.")

    chosen    = pairs[PAIR_RANK - 1]
    security1 = chosen["s1"]
    security2 = chosen["s2"]
    print(f"Using pair: {security1}/{security2}  (avg R²={chosen['r2']:.4f})\n")

    prices1 = df[security1].values.astype(float)
    prices2 = df[security2].values.astype(float)

    print(f"{'N':>6}  {'windows':>8}  {'mean Sharpe':>12}  {'std Sharpe':>11}  {'median':>8}")
    print("─" * 52)

    results = []
    for N in WINDOW_CANDIDATES:
        if N + TEST_WINDOW > len(prices1):
            print(f"{N:>6}  (skipped — not enough data)")
            continue

        sharpes = evaluate_window(prices1, prices2, N)
        if not sharpes:
            results.append((N, 0, np.nan, np.nan, np.nan))
            print(f"{N:>6}  {'0':>8}  {'—':>12}")
            continue

        arr    = np.array(sharpes)
        mean_s = arr.mean()
        std_s  = arr.std()
        med_s  = np.median(arr)

        results.append((N, len(sharpes), mean_s, std_s, med_s))
        print(f"{N:>6}  {len(sharpes):>8}  {mean_s:>12.4f}  {std_s:>11.4f}  {med_s:>8.4f}")

    # ── Rank by mean Sharpe ───────────────────────────────────────────────────
    valid = [(n, cnt, m, s, med) for n, cnt, m, s, med in results if not np.isnan(m)]
    valid.sort(key=lambda x: x[2], reverse=True)

    print()
    print("=" * 52)
    print("  Ranked by mean Sharpe")
    print("=" * 52)
    for rank, (n, cnt, m, s, med) in enumerate(valid[:10], 1):
        print(f"  #{rank:>2}  N={n:>4}  mean={m:>8.4f}  std={s:>7.4f}  median={med:>7.4f}")

    best_N = valid[0][0] if valid else None
    print(f"\nBest lookback window: N = {best_N}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ns     = [r[0] for r in results if not np.isnan(r[2])]
    means  = [r[2] for r in results if not np.isnan(r[2])]
    stds   = [r[3] for r in results if not np.isnan(r[2])]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(ns, means, width=max(WINDOW_CANDIDATES) / len(ns) * 0.8,
                  color=["#2ecc71" if m == max(means) else "#3498db" for m in means],
                  alpha=0.85)
    ax.errorbar(ns, means, yerr=stds, fmt="none", color="black", capsize=3, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lookback window (ticks)")
    ax.set_ylabel("Mean out-of-sample Sharpe")
    ax.set_title(f"Lookback window optimisation — {security1}/{security2}\n"
                 f"TEST_WINDOW={TEST_WINDOW} ticks  |  STEP={STEP}  |  best N={best_N}")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], rotation=45, ha="right")

    if best_N is not None:
        best_idx = ns.index(best_N)
        ax.annotate(f"best: {best_N}",
                    xy=(best_N, means[best_idx]),
                    xytext=(best_N, means[best_idx] + stds[best_idx] + 0.05),
                    ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    plt.show()

    return results, best_N


if __name__ == "__main__":
    results, best_N = run()

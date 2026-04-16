"""
Threshold Optimizer — Percentage / Ratio Model
================================================
Same grid-search as threshold_optimizer.py but uses the ratio spread:

    spread = y/x − ratio

instead of the absolute spread  y − (coef·x + intercept).

Usage
-----
    python threshold_optimizer_pct.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from daily_pair_regression_pct import run as get_daily_pairs

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = os.path.join(os.path.dirname(__file__), "../Competition/Alpha Testing V2/data.csv")

# 1 = best pair, 2 = second best, etc.  Set None to use manual values below.
PAIR_RANK  = 1

# Manual override — only used when PAIR_RANK is None
SECURITY1  = "IND"
SECURITY2  = "ETF"
RATIO      = 2.2650   # avg(security2 / security1)

TRAIN_DAYS = 30
CAPITAL    = 1_000_000
GRID_STEPS = 20
# ─────────────────────────────────────────────────────────────────────────────


def backtest(df, spread, sd, security1, security2, buy_in, back):
    entry_thresh = buy_in * sd
    exit_thresh  = back   * sd

    capital     = CAPITAL
    total       = capital
    per         = capital / 2
    tot_sec2    = 0.0
    tot_sec1    = 0.0
    in_position = False
    profit      = [1.0]

    idx = df.index.tolist()
    for k, i in enumerate(idx):
        if k == 0:
            continue

        s = spread[i]

        if s >= entry_thresh and not in_position:
            qty2 = per // df[security2][i]
            qty1 = per // df[security1][i]
            tot_sec2  -= qty2
            tot_sec1  += qty1
            total     += qty2 * df[security2][i]
            total     -= qty1 * df[security1][i]
            in_position = True

        elif s <= exit_thresh and in_position:
            total    -= tot_sec2 * df[security2][i]
            total    += tot_sec1 * df[security1][i]
            tot_sec2  = 0.0
            tot_sec1  = 0.0
            in_position = False

        profit.append(total / capital)

    profit_series = pd.Series(profit)
    sigma = float(profit_series.std())
    if sigma == 0:
        return -np.inf

    Rp = total / capital
    Rf = df["IND"].iloc[-1] / df["IND"].iloc[0]
    return float((Rp - Rf) / sigma)


def find_best_thresholds_pct(df, security1, security2, ratio, train_days=30, grid_steps=20):
    """
    Programmatic entry point — returns (buy_in, back, sd, best_sharpe).
    Spread = security2/security1 − ratio.
    """
    train_df   = df[df["day"] <= train_days]
    ratio_series = train_df[security2].values / train_df[security1].values
    sd           = float((ratio_series - ratio).std())

    test_df = df[df["day"] > train_days].copy()
    spread  = test_df[security2] / test_df[security1] - ratio

    thresholds = np.linspace(0, 2, grid_steps)
    results    = np.full((grid_steps, grid_steps), np.nan)

    for i, buy_in in enumerate(thresholds):
        for j, back in enumerate(thresholds):
            if back >= buy_in:
                continue
            results[i, j] = backtest(test_df, spread, sd, security1, security2, buy_in, back)

    flat_best      = np.nanargmax(results)
    best_i, best_j = np.unravel_index(flat_best, results.shape)
    return float(thresholds[best_i]), float(thresholds[best_j]), sd, float(results[best_i, best_j])


def run():
    df = pd.read_csv(DATA_FILE)

    # ── Resolve pair ──────────────────────────────────────────────────────────
    if PAIR_RANK is not None:
        print(f"Running daily ratio regression to find rank #{PAIR_RANK} pair...\n")
        _, _, pairs = get_daily_pairs()
        if PAIR_RANK > len(pairs):
            raise ValueError(f"PAIR_RANK={PAIR_RANK} but only {len(pairs)} pairs found.")
        chosen    = pairs[PAIR_RANK - 1]
        security1 = chosen["s1"]
        security2 = chosen["s2"]
        ratio     = chosen["ratio"]
        print(f"\nUsing rank #{PAIR_RANK} pair: {security1}/{security2}  "
              f"(avg R²={chosen['r2']:.4f}  ratio={ratio:.6f})\n")
    else:
        security1 = SECURITY1
        security2 = SECURITY2
        ratio     = RATIO

    # ── Compute SD from training spread ───────────────────────────────────────
    train_df     = df[df["day"] <= TRAIN_DAYS]
    ratio_series = train_df[security2].values / train_df[security1].values
    sd           = float((ratio_series - ratio).std())

    test_df = df[df["day"] > TRAIN_DAYS].copy()
    spread  = test_df[security2] / test_df[security1] - ratio

    print(f"Pair:  {security2}/{security1}  ratio={ratio:.6f}")
    print(f"SD:    {sd:.6f}")
    print(f"Test rows: {len(test_df)}  (days > {TRAIN_DAYS})\n")

    thresholds = np.linspace(0, 2, GRID_STEPS)
    results    = np.full((GRID_STEPS, GRID_STEPS), np.nan)

    for i, buy_in in enumerate(thresholds):
        for j, back in enumerate(thresholds):
            if back >= buy_in:
                continue
            results[i, j] = backtest(test_df, spread, sd, security1, security2, buy_in, back)

    # ── Find best ─────────────────────────────────────────────────────────────
    flat_best = np.nanargmax(results)
    best_i, best_j = np.unravel_index(flat_best, results.shape)
    best_buy_in    = thresholds[best_i]
    best_back      = thresholds[best_j]
    best_sharpe    = results[best_i, best_j]

    print("=" * 45)
    print("  Best thresholds")
    print("=" * 45)
    print(f"  buy_in  : {best_buy_in:.4f}  (= {best_buy_in * sd:.6f} ratio units)")
    print(f"  back    : {best_back:.4f}  (= {best_back * sd:.6f} ratio units)")
    print(f"  Sharpe  : {best_sharpe:.4f}")
    print("=" * 45)

    print("\nTop 10 combinations:")
    print(f"  {'buy_in':>8}  {'back':>8}  {'sharpe':>10}")
    print("  " + "-" * 32)
    flat_sorted = np.argsort(results.flatten())[::-1]
    shown = 0
    for idx in flat_sorted:
        if shown >= 10:
            break
        ii, jj = np.unravel_index(idx, results.shape)
        v = results[ii, jj]
        if np.isnan(v):
            continue
        print(f"  {thresholds[ii]:>8.4f}  {thresholds[jj]:>8.4f}  {v:>10.4f}")
        shown += 1

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(results, origin="lower", aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Sharpe ratio")

    tick_labels = [f"{t:.2f}" for t in thresholds]
    step = max(1, GRID_STEPS // 10)
    ax.set_xticks(range(0, GRID_STEPS, step))
    ax.set_xticklabels(tick_labels[::step], rotation=45, ha="right")
    ax.set_yticks(range(0, GRID_STEPS, step))
    ax.set_yticklabels(tick_labels[::step])
    ax.set_xlabel("back  (exit threshold × SD)")
    ax.set_ylabel("buy_in  (entry threshold × SD)")
    ax.set_title(f"Sharpe ratio grid — {security2}/{security1}  (ratio spread)\n"
                 f"Best: buy_in={best_buy_in:.3f}, back={best_back:.3f}  "
                 f"(Sharpe={best_sharpe:.3f})")

    ax.plot(best_j, best_i, marker="*", color="white", markersize=14)
    plt.tight_layout()
    plt.show()

    return best_buy_in, best_back, best_sharpe


if __name__ == "__main__":
    run()

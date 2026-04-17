"""
Backtest — Together Strategy
==============================
Simulates together.py on the historical data in together/data.csv.
Runs both strategies tick-by-tick and plots their individual and combined PnL.

Usage
-----
    python test_together.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_DR     = os.path.join(_HERE, "../Daily Regression")
_ROTMAN = os.path.join(_HERE, "../Rotman")
sys.path.insert(0, _HERE)
sys.path.insert(0, _DR)
sys.path.insert(0, _ROTMAN)

from daily_pair_regression import run as get_daily_pairs

# ── Config (must match together.py) ──────────────────────────────────────────
DATA_FILE      = os.path.join(_HERE, "data.csv")
TOTAL_CAPITAL  = 20_000_000
HALF_CAPITAL   = TOTAL_CAPITAL // 2
TRADE_FRACTION = 0.4
LOOKBACK       = 40
REFIT_EVERY    = 5
PAIR_RANK      = 1
B_SEC1         = "IND"
B_SEC2         = "ETF"
# ─────────────────────────────────────────────────────────────────────────────


def _calibrate_thresholds(spread_arr, sd, grid_steps=25):
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


def _fit_model(buf1, buf2):
    h1  = np.array(buf1[-LOOKBACK:])
    h2  = np.array(buf2[-LOOKBACK:])
    mdl = LinearRegression(fit_intercept=True).fit(h1.reshape(-1, 1), h2)
    c   = float(mdl.coef_[0])
    ic  = float(mdl.intercept_)
    res = h2 - (c * h1 + ic)
    sd  = float(res.std())
    en, ex = _calibrate_thresholds(res, sd)
    return c, ic, en, ex


def simulate(prices1: np.ndarray, prices2: np.ndarray, capital: float, trade_fraction: float):
    """
    Simulate one strategy on two price series.
    Returns a numpy array of cumulative realized PnL at each tick.
    """
    notional = capital * trade_fraction
    buf1, buf2 = [], []

    coef, intercept, entry_thresh, exit_thresh = 0.0, 0.0, 1.0, 0.3
    in_position = False
    tot1 = 0
    tot2 = 0
    entry_p1 = 0.0
    entry_p2 = 0.0
    realized_pnl = 0.0
    pnl_series = []

    for i, (p1, p2) in enumerate(zip(prices1, prices2)):
        buf1.append(p1)
        buf2.append(p2)

        # Refit every REFIT_EVERY ticks once we have enough data
        if len(buf1) >= LOOKBACK and (i + 1) % REFIT_EVERY == 0:
            coef, intercept, entry_thresh, exit_thresh = _fit_model(buf1, buf2)

        # Skip trading until we have enough history for the first fit
        if len(buf1) < LOOKBACK:
            pnl_series.append(realized_pnl)
            continue

        spread = p2 - (coef * p1 + intercept)

        # Unrealized PnL
        if in_position:
            if tot2 < 0:
                unrealized = abs(tot2) * (entry_p2 - p2) + abs(tot1) * (p1 - entry_p1)
            else:
                unrealized = abs(tot2) * (p2 - entry_p2) + abs(tot1) * (entry_p1 - p1)
        else:
            unrealized = 0.0

        pnl_series.append(realized_pnl + unrealized)

        qty2 = int(notional // p2)
        qty1 = int(notional // p1)

        if not in_position:
            if spread >= entry_thresh:
                tot2 = -qty2; tot1 = qty1
                entry_p1 = p1; entry_p2 = p2
                in_position = True
            elif spread <= -entry_thresh:
                tot2 = qty2; tot1 = -qty1
                entry_p1 = p1; entry_p2 = p2
                in_position = True
        else:
            if tot2 < 0 and spread <= exit_thresh:
                realized_pnl += abs(tot2) * (entry_p2 - p2) + abs(tot1) * (p1 - entry_p1)
                tot2 = 0; tot1 = 0; in_position = False
            elif tot2 > 0 and spread >= -exit_thresh:
                realized_pnl += abs(tot2) * (p2 - entry_p2) + abs(tot1) * (entry_p1 - p1)
                tot2 = 0; tot1 = 0; in_position = False

    # Close any open position at end
    if in_position:
        p1, p2 = prices1[-1], prices2[-1]
        if tot2 < 0:
            realized_pnl += abs(tot2) * (entry_p2 - p2) + abs(tot1) * (p1 - entry_p1)
        else:
            realized_pnl += abs(tot2) * (p2 - entry_p2) + abs(tot1) * (entry_p1 - p1)
        pnl_series[-1] = realized_pnl

    return np.array(pnl_series)


def run():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows.  Columns: {list(df.columns)}")

    # ── Pick Strategy A pair ──────────────────────────────────────────────────
    print(f"\nRunning daily pair regression — picking rank #{PAIR_RANK}...")
    _, _, _, pairs = get_daily_pairs()
    if PAIR_RANK > len(pairs):
        raise ValueError(f"PAIR_RANK={PAIR_RANK} but only {len(pairs)} pairs found.")
    chosen = pairs[PAIR_RANK - 1]
    a_sec1 = chosen["s1"]
    a_sec2 = chosen["s2"]
    print(f"Strategy A pair: {a_sec1} / {a_sec2}  (avg R²={chosen['r2']:.4f})")
    print(f"Strategy B pair: {B_SEC1} / {B_SEC2}  (fixed)")

    # ── Extract price series ──────────────────────────────────────────────────
    a_prices1 = df[a_sec1].values.astype(float)
    a_prices2 = df[a_sec2].values.astype(float)
    b_prices1 = df[B_SEC1].values.astype(float)
    b_prices2 = df[B_SEC2].values.astype(float)

    # ── Simulate ──────────────────────────────────────────────────────────────
    print("\nSimulating Strategy A...")
    pnl_a = simulate(a_prices1, a_prices2, HALF_CAPITAL, TRADE_FRACTION)

    print("Simulating Strategy B...")
    pnl_b = simulate(b_prices1, b_prices2, HALF_CAPITAL, TRADE_FRACTION)

    pnl_combined = pnl_a + pnl_b
    ticks = np.arange(len(pnl_a))

    # ── Stats ──────────────────────────────────────────────────────────────────
    def sharpe(pnl):
        d = np.diff(pnl)
        return d.mean() / (d.std() + 1e-9) if len(d) > 1 else 0.0

    print(f"\n{'':=<55}")
    print(f"  Strategy A  ({a_sec1}/{a_sec2})")
    print(f"    Final PnL : {pnl_a[-1]:>+15,.0f}")
    print(f"    Sharpe    : {sharpe(pnl_a):>+10.4f}")
    print(f"  Strategy B  ({B_SEC1}/{B_SEC2})")
    print(f"    Final PnL : {pnl_b[-1]:>+15,.0f}")
    print(f"    Sharpe    : {sharpe(pnl_b):>+10.4f}")
    print(f"  Combined")
    print(f"    Final PnL : {pnl_combined[-1]:>+15,.0f}")
    print(f"    Sharpe    : {sharpe(pnl_combined):>+10.4f}")
    print(f"{'':=<55}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Together Backtest  —  Capital: ${TOTAL_CAPITAL:,.0f}  "
        f"(${HALF_CAPITAL:,.0f} each)  |  Refit every {REFIT_EVERY} ticks",
        fontsize=13,
    )

    axes[0].plot(ticks, pnl_a, color="#3498db", linewidth=1)
    axes[0].axhline(0, color="gray", linewidth=0.7, linestyle="--")
    axes[0].fill_between(ticks, pnl_a, 0,
                         where=(pnl_a >= 0), alpha=0.15, color="#2ecc71")
    axes[0].fill_between(ticks, pnl_a, 0,
                         where=(pnl_a < 0),  alpha=0.15, color="#e74c3c")
    axes[0].set_ylabel("PnL ($)")
    axes[0].set_title(
        f"Strategy A — {a_sec1}/{a_sec2}  |  "
        f"Final: {pnl_a[-1]:+,.0f}  Sharpe: {sharpe(pnl_a):.3f}"
    )

    axes[1].plot(ticks, pnl_b, color="#9b59b6", linewidth=1)
    axes[1].axhline(0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].fill_between(ticks, pnl_b, 0,
                         where=(pnl_b >= 0), alpha=0.15, color="#2ecc71")
    axes[1].fill_between(ticks, pnl_b, 0,
                         where=(pnl_b < 0),  alpha=0.15, color="#e74c3c")
    axes[1].set_ylabel("PnL ($)")
    axes[1].set_title(
        f"Strategy B — {B_SEC1}/{B_SEC2}  |  "
        f"Final: {pnl_b[-1]:+,.0f}  Sharpe: {sharpe(pnl_b):.3f}"
    )

    axes[2].plot(ticks, pnl_combined, color="#e67e22", linewidth=1.2)
    axes[2].axhline(0, color="gray", linewidth=0.7, linestyle="--")
    axes[2].fill_between(ticks, pnl_combined, 0,
                         where=(pnl_combined >= 0), alpha=0.15, color="#2ecc71")
    axes[2].fill_between(ticks, pnl_combined, 0,
                         where=(pnl_combined < 0),  alpha=0.15, color="#e74c3c")
    axes[2].set_ylabel("PnL ($)")
    axes[2].set_xlabel("Tick")
    axes[2].set_title(
        f"Combined  |  "
        f"Final: {pnl_combined[-1]:+,.0f}  Sharpe: {sharpe(pnl_combined):.3f}"
    )

    for ax in axes:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()

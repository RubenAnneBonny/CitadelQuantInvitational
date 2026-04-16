"""
Daily Pair Regression
======================
For every pair of stocks, fits a linear regression on each individual day,
then averages the R² across all days and outputs a summary matrix + heatmap.

Usage
-----
    python daily_pair_regression.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "../Competition/Alpha Testing V2/data.csv")
STOCKS    = ["AAA", "BBB", "CCC", "DDD", "ETF", "IND"]
# ─────────────────────────────────────────────────────────────────────────────


def regression_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Fit OLS y ~ coef*x + intercept and return R²."""
    model     = LinearRegression(fit_intercept=True).fit(x.reshape(-1, 1), y)
    y_hat     = model.predict(x.reshape(-1, 1))
    ss_res    = float(((y - y_hat) ** 2).sum())
    ss_tot    = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def run():
    df   = pd.read_csv(DATA_FILE)
    days = sorted(df["day"].unique())

    print(f"Loaded {len(df)} rows across {len(days)} days.")
    print(f"Stocks: {STOCKS}\n")

    # accumulators over all days
    r2_sums        = {s1: {s2: 0.0 for s2 in STOCKS} for s1 in STOCKS}
    coef_sums      = {s1: {s2: 0.0 for s2 in STOCKS} for s1 in STOCKS}
    intercept_sums = {s1: {s2: 0.0 for s2 in STOCKS} for s1 in STOCKS}
    n_days         = len(days)

    for day in days:
        day_df = df[df["day"] == day]

        for i, s1 in enumerate(STOCKS):
            for j, s2 in enumerate(STOCKS):
                if s1 == s2:
                    continue

                x = day_df[s1].values
                y = day_df[s2].values

                model     = LinearRegression(fit_intercept=True).fit(x.reshape(-1, 1), y)
                coef      = float(model.coef_[0])
                intercept = float(model.intercept_)

                y_hat  = model.predict(x.reshape(-1, 1))
                ss_res = float(((y - y_hat) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                r2_sums[s1][s2]        += r2
                coef_sums[s1][s2]      += coef
                intercept_sums[s1][s2] += intercept

    # Average over days
    avg_r2        = pd.DataFrame(
        {s1: {s2: (r2_sums[s1][s2] / n_days if s1 != s2 else 1.0) for s2 in STOCKS}
         for s1 in STOCKS}
    ).T

    avg_coef      = pd.DataFrame(
        {s1: {s2: (coef_sums[s1][s2] / n_days if s1 != s2 else 1.0) for s2 in STOCKS}
         for s1 in STOCKS}
    ).T

    avg_intercept = pd.DataFrame(
        {s1: {s2: (intercept_sums[s1][s2] / n_days if s1 != s2 else 0.0) for s2 in STOCKS}
         for s1 in STOCKS}
    ).T

    # ── Ranked pairs ──────────────────────────────────────────────────────────
    pairs = []
    for i, s1 in enumerate(STOCKS):
        for j, s2 in enumerate(STOCKS):
            if j <= i:
                continue
            r2 = (avg_r2.loc[s1, s2] + avg_r2.loc[s2, s1]) / 2
            pairs.append({
                "r2":        r2,
                "s1":        s1,
                "s2":        s2,
                "coef":      float(avg_coef.loc[s1, s2]),
                "intercept": float(avg_intercept.loc[s1, s2]),
            })
    pairs.sort(key=lambda p: p["r2"], reverse=True)

    # ── Print results ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Average R²  (row = x, col = y,  i.e. y ~ coef·x + intercept)")
    print("=" * 60)
    print(avg_r2.round(4).to_string())

    print()
    print("=" * 60)
    print("Average regression coefficient  (y ~ coef·x + intercept)")
    print("=" * 60)
    print(avg_coef.round(4).to_string())

    print()
    print("=" * 60)
    print("Pairs ranked by average R²")
    print("=" * 60)
    for rank, p in enumerate(pairs, 1):
        print(f"  #{rank:>2}  {p['s1']}/{p['s2']:<6}  R²={p['r2']:.4f}  "
              f"coef={p['coef']:.4f}  intercept={p['intercept']:.4f}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Daily Linear Regression — Averaged over {n_days} days", fontsize=12)

    for ax, data, title in zip(
        axes,
        [avg_r2, avg_coef],
        ["Average R²", "Average Coefficient"]
    ):
        vals = data.values.astype(float)
        im   = ax.imshow(vals, cmap="YlGn", aspect="auto",
                         vmin=0 if "R²" in title else None)
        ax.set_xticks(range(len(STOCKS))); ax.set_xticklabels(STOCKS, rotation=45, ha="right")
        ax.set_yticks(range(len(STOCKS))); ax.set_yticklabels(STOCKS)
        ax.set_xlabel("y  (dependent)");  ax.set_ylabel("x  (independent)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for r in range(len(STOCKS)):
            for c in range(len(STOCKS)):
                ax.text(c, r, f"{vals[r, c]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.tight_layout()
    plt.show()

    return avg_r2, avg_coef, avg_intercept, pairs


if __name__ == "__main__":
    avg_r2, avg_coef, avg_intercept, pairs = run()

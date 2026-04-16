"""
Pair Cointegration & Backtest
==============================
Functionalised version of Cointegration_test.ipynb + Backtest.ipynb.
Test any set of stocks for cointegration and run mean-reversion backtests.

Quick start
-----------
    from pair_analysis import run_pairs

    df = pd.read_csv("data.csv")

    # Test all pairs, train on first 30 days, backtest on the rest
    summary = run_pairs(df, ["AAA", "BBB", "CCC", "DDD", "ETF", "IND"], train_days=30)

    # Or drill into one specific pair with plots
    result = run_pair(df, stock_x="IND", stock_y="ETF", train_days=30, plot=True)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────────────────────────────────────────
# Quick linear fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_linear(x, y, plot=True):
    """
    Fit a linear regression between two series and print/plot the result.

    Parameters
    ----------
    x, y  : array-like or pd.Series — the two series to fit (y ~ coef*x + intercept).
    plot  : if True, show a scatter + fit line and a residual plot.

    Returns
    -------
    dict with intercept, coef, spread_std, r_squared.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    model     = LinearRegression(fit_intercept=True).fit(x.reshape(-1, 1), y)
    intercept = float(model.intercept_)
    coef      = float(model.coef_[0])
    y_hat     = coef * x + intercept
    residuals = y - y_hat
    spread_std = float(residuals.std())
    ss_res    = float((residuals ** 2).sum())
    ss_tot    = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print("=" * 40)
    print("  Linear fit:  y = coef·x + intercept")
    print("=" * 40)
    print(f"  intercept  : {intercept:.6f}")
    print(f"  coef       : {coef:.6f}")
    print(f"  spread_std : {spread_std:.6f}")
    print(f"  R²         : {r_squared:.6f}")
    print("=" * 40)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Scatter + fit line
        axes[0].scatter(x, y, s=2, alpha=0.3, color="#888888", label="data")
        x_line = np.linspace(x.min(), x.max(), 200)
        axes[0].plot(x_line, coef * x_line + intercept,
                     color="#3ddc84", linewidth=1.5, label=f"y = {coef:.4f}·x + {intercept:.4f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Scatter  (R² = {r_squared:.4f})")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.2)

        # Residuals over time
        axes[1].plot(residuals, color="#aaaaaa", linewidth=0.7, alpha=0.8)
        axes[1].axhline(0,              color="#555555", linewidth=0.8, linestyle="--")
        axes[1].axhline( spread_std,    color="#ffc44d", linewidth=0.8, linestyle=":", label="+1σ")
        axes[1].axhline(-spread_std,    color="#ffc44d", linewidth=0.8, linestyle=":", label="-1σ")
        axes[1].fill_between(range(len(residuals)), residuals, 0,
                             where=(residuals >= 0), color="#ff5c5c", alpha=0.2)
        axes[1].fill_between(range(len(residuals)), residuals, 0,
                             where=(residuals <  0), color="#3ddc84", alpha=0.2)
        axes[1].set_xlabel("Observation")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals (spread)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()

    return {"intercept": intercept, "coef": coef,
            "spread_std": spread_std, "r_squared": r_squared}


# ─────────────────────────────────────────────────────────────────────────────
# Cointegration
# ─────────────────────────────────────────────────────────────────────────────

def find_cointegrated_pairs(df, stocks, train_days=None, confidence="95%"):
    """
    Johansen trace test on every unique pair in `stocks`.

    Parameters
    ----------
    df          : DataFrame with a 'day' column and one column per stock.
    stocks      : list of stock names to consider.
    train_days  : if set, only use rows where df['day'] <= train_days.
    confidence  : '90%', '95%', or '99%'.

    Returns
    -------
    List of (trace_stat, (stock1, stock2)) sorted descending by trace_stat.
    Only pairs that exceed the critical value are included.
    """
    conf_idx = {"90%": 0, "95%": 1, "99%": 2}[confidence]
    data = df[df["day"] <= train_days] if train_days is not None else df

    pairs = []
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            s1, s2 = stocks[i], stocks[j]
            subset = data[[s1, s2]].dropna()
            cj = coint_johansen(subset, det_order=1, k_ar_diff=1)
            if cj.trace_stat[0] > cj.trace_stat_crit_vals[0][conf_idx]:
                pairs.append((float(cj.trace_stat[0]), (s1, s2)))

    pairs.sort(reverse=True)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Hedge ratio
# ─────────────────────────────────────────────────────────────────────────────

def fit_ols(df, stock_x, stock_y, train_days=None):
    """
    OLS regression: stock_y ~ coef * stock_x + intercept

    Returns
    -------
    (intercept, coef, spread_std)
        intercept   : float
        coef        : float
        spread_std  : float — std dev of the residuals on the training data
    """
    data = df[df["day"] <= train_days] if train_days is not None else df
    X = data[stock_x].values.reshape(-1, 1)
    y = data[stock_y].values
    model = LinearRegression(fit_intercept=True).fit(X, y)
    intercept = float(model.intercept_)
    coef      = float(model.coef_[0])
    residuals = y - (coef * data[stock_x].values + intercept)
    return intercept, coef, float(residuals.std())


# ─────────────────────────────────────────────────────────────────────────────
# Spread
# ─────────────────────────────────────────────────────────────────────────────

def compute_spread(df, stock_x, stock_y, intercept, coef):
    """spread = stock_y − (coef * stock_x + intercept)"""
    return df[stock_y] - (coef * df[stock_x] + intercept)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

def backtest_pair(df, stock_x, stock_y, intercept, coef, spread_std,
                  entry_z=1.0, exit_z=0.5, capital=1_000_000, two_sided=True):
    """
    Mean-reversion backtest on the spread = stock_y − (coef * stock_x + intercept).

    Logic
    -----
    - Spread > +entry_z * std  →  short stock_y  (spread too high, expect reversion down)
    - Spread < -entry_z * std  →  long  stock_y  (spread too low,  expect reversion up)
      (long side only active when two_sided=True)
    - Exit when |spread| < exit_z * std

    Parameters
    ----------
    df                      : test DataFrame (rows to trade on).
    stock_x, stock_y        : the pair.
    intercept, coef         : from fit_ols (trained on training data).
    spread_std              : std dev of the spread from training data.
    entry_z, exit_z         : z-score thresholds.
    capital                 : notional capital.
    two_sided               : if True, also trade long when spread is very negative.

    Returns
    -------
    dict with profit_series, final_value, total_return_pct, sharpe, open_position.
    """
    spread         = compute_spread(df, stock_x, stock_y, intercept, coef)
    entry_hi       =  entry_z * spread_std
    entry_lo       = -entry_z * spread_std
    exit_hi        =  exit_z  * spread_std
    exit_lo        = -exit_z  * spread_std
    trade_notional = capital / 4

    total      = capital
    position_y = 0.0        # shares of stock_y  (negative = short)
    profit     = []

    idx = df.index.tolist()
    for k, i in enumerate(idx):
        if k == 0:
            profit.append(total / capital)
            continue

        prev = idx[k - 1]
        price = df[stock_y][i]

        # ── Enter short: spread crossed above entry_hi ────────────────────────
        if spread[i] >= entry_hi and spread[prev] < entry_hi and position_y == 0.0:
            shares      = trade_notional / price
            position_y -= shares
            total      += shares * price

        # ── Exit short: spread fell back below exit_hi ────────────────────────
        if spread[i] <= exit_hi and spread[prev] > exit_hi and position_y < 0.0:
            total      -= position_y * price   # position_y is negative → adds to total
            position_y  = 0.0

        if two_sided:
            # ── Enter long: spread crossed below entry_lo ─────────────────────
            if spread[i] <= entry_lo and spread[prev] > entry_lo and position_y == 0.0:
                shares      = trade_notional / price
                position_y += shares
                total      -= shares * price

            # ── Exit long: spread rose back above exit_lo ─────────────────────
            if spread[i] >= exit_lo and spread[prev] < exit_lo and position_y > 0.0:
                total      -= position_y * price
                position_y  = 0.0

        profit.append(total / capital)

    profit_series = pd.Series(profit, index=df.index)

    Rp    = total / capital
    Rf    = df["IND"].iloc[-1] / df["IND"].iloc[0] if "IND" in df.columns else 1.0
    sigma = float(profit_series.std())
    sharpe = float((Rp - Rf) / sigma) if sigma > 0 else np.nan

    return {
        "profit_series":    profit_series,
        "final_value":      total,
        "total_return_pct": (total / capital - 1) * 100,
        "sharpe":           sharpe,
        "open_position":    position_y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-pair pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pair(df, stock_x, stock_y, train_days=30,
             entry_z=1.0, exit_z=0.5, capital=1_000_000,
             two_sided=True, plot=True):
    """
    Full pipeline for one pair: fit OLS on training days, backtest on the rest.

    Parameters
    ----------
    df          : full DataFrame (train + test rows).
    stock_x     : independent variable in the regression.
    stock_y     : dependent variable (the one being traded).
    train_days  : rows with df['day'] <= train_days are training data.
    entry_z     : spread z-score to open a position.
    exit_z      : spread z-score to close a position.
    capital     : notional capital.
    two_sided   : trade both long and short.
    plot        : if True, show a 3-panel chart.

    Returns
    -------
    dict with intercept, coef, spread_std_train, backtest metrics, profit_series.
    """
    train = df[df["day"] <= train_days]
    test  = df[df["day"] >  train_days]

    intercept, coef, spread_std = fit_ols(train, stock_x, stock_y)

    bt = backtest_pair(test, stock_x, stock_y, intercept, coef, spread_std,
                       entry_z=entry_z, exit_z=exit_z, capital=capital,
                       two_sided=two_sided)

    if plot:
        _plot_pair(test, stock_x, stock_y, intercept, coef, spread_std,
                   bt, entry_z, exit_z)

    return {
        "pair":             (stock_x, stock_y),
        "intercept":        intercept,
        "coef":             coef,
        "spread_std_train": spread_std,
        **{k: v for k, v in bt.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-pair pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pairs(df, stocks, train_days=30, entry_z=1.0, exit_z=0.5,
              capital=1_000_000, two_sided=True, confidence="95%",
              plot_each=False):
    """
    Find all cointegrated pairs among `stocks`, fit hedge ratios, and backtest each.

    Parameters
    ----------
    df          : full DataFrame.
    stocks      : list of stock names, e.g. ["AAA", "BBB", "ETF", "IND"].
    train_days  : rows with df['day'] <= train_days are used for fitting.
    entry_z     : z-score to open a position.
    exit_z      : z-score to close a position.
    capital     : notional capital per pair.
    two_sided   : trade both directions.
    confidence  : Johansen test threshold — '90%', '95%', or '99%'.
    plot_each   : if True, produce a chart for every pair.

    Returns
    -------
    pd.DataFrame with one row per cointegrated pair, sorted by Sharpe ratio.
    """
    pairs = find_cointegrated_pairs(df, stocks, train_days=train_days,
                                    confidence=confidence)
    if not pairs:
        print("No cointegrated pairs found.")
        return pd.DataFrame()

    print(f"Found {len(pairs)} cointegrated pair(s) at {confidence}:\n")
    for stat, (s1, s2) in pairs:
        print(f"  {s1}/{s2}   trace_stat = {stat:.2f}")

    rows = []
    for _, (s1, s2) in pairs:
        result = run_pair(df, s1, s2, train_days=train_days,
                          entry_z=entry_z, exit_z=exit_z, capital=capital,
                          two_sided=two_sided, plot=plot_each)
        rows.append({
            "pair":             f"{s1}/{s2}",
            "coef":             round(result["coef"], 4),
            "intercept":        round(result["intercept"], 4),
            "spread_std_train": round(result["spread_std_train"], 4),
            "total_return_%":   round(result["total_return_pct"], 2),
            "sharpe":           round(result["sharpe"], 4),
            "final_value":      round(result["final_value"], 2),
            "open_position":    round(result["open_position"], 4),
        })

    summary = (pd.DataFrame(rows)
               .sort_values("sharpe", ascending=False)
               .reset_index(drop=True))

    print("\n" + "=" * 65)
    print("SUMMARY  (sorted by Sharpe)")
    print("=" * 65)
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pair(test_df, stock_x, stock_y, intercept, coef, spread_std,
               bt, entry_z, exit_z):
    spread = compute_spread(test_df, stock_x, stock_y, intercept, coef)
    entry_hi =  entry_z * spread_std
    exit_hi  =  exit_z  * spread_std
    entry_lo = -entry_z * spread_std
    exit_lo  = -exit_z  * spread_std

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Pair: {stock_y} ~ {coef:.4f}·{stock_x} + {intercept:.4f}  "
        f"(entry ±{entry_z}σ, exit ±{exit_z}σ)",
        fontsize=11, fontweight="bold"
    )

    # Panel 1 — prices
    axes[0].plot(test_df.index, test_df[stock_y], linewidth=0.8, label=stock_y)
    axes[0].plot(test_df.index, coef * test_df[stock_x] + intercept,
                 linewidth=0.8, linestyle="--", label=f"fitted ({stock_x})", alpha=0.7)
    axes[0].set_ylabel("Price")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Panel 2 — spread with thresholds
    axes[1].plot(test_df.index, spread, color="#aaaaaa", linewidth=0.8)
    axes[1].axhline(0,         color="#555555", linewidth=0.8, linestyle="--")
    axes[1].axhline(entry_hi,  color="#ff5c5c", linewidth=0.8, linestyle=":", label=f"+{entry_z}σ entry")
    axes[1].axhline(exit_hi,   color="#ffc44d", linewidth=0.8, linestyle=":", label=f"+{exit_z}σ exit")
    axes[1].axhline(entry_lo,  color="#3ddc84", linewidth=0.8, linestyle=":", label=f"-{entry_z}σ entry")
    axes[1].axhline(exit_lo,   color="#ffc44d", linewidth=0.8, linestyle=":")
    axes[1].fill_between(test_df.index, spread, 0,
                         where=(spread >= 0), color="#ff5c5c", alpha=0.15)
    axes[1].fill_between(test_df.index, spread, 0,
                         where=(spread <  0), color="#3ddc84", alpha=0.15)
    axes[1].set_ylabel("Spread")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    # Panel 3 — cumulative return
    ps = bt["profit_series"]
    axes[2].plot(ps.index, ps.values, color="#3ddc84", linewidth=1.0)
    axes[2].axhline(1, color="#555555", linewidth=0.8, linestyle="--")
    axes[2].fill_between(ps.index, ps.values, 1,
                         where=(ps.values >= 1), color="#3ddc84", alpha=0.2)
    axes[2].fill_between(ps.index, ps.values, 1,
                         where=(ps.values <  1), color="#ff5c5c", alpha=0.2)
    axes[2].set_ylabel("Portfolio (× capital)")
    axes[2].set_xlabel("Tick")
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(data_path)

    stocks = ["AAA", "BBB", "CCC", "DDD", "ETF", "IND"]
    # summary = run_pairs(df, stocks, train_days=30, plot_each=True)

    fit_linear(df["CCC"], df["ETF"])

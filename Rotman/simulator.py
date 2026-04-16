"""
Trade Simulator
===============
Simulates a strategy driven by alpha signals in [-1, 1] and computes PnL
and Sharpe ratio.

Alpha interpretation
--------------------
  +1  → fully long  (hold `capital` dollars of the stock)
   0  → flat        (no position)
  -1  → fully short (short `capital` dollars of the stock)
  Fractional values scale the position linearly.

The position at tick t determines the P&L earned as prices move from tick t
to tick t+1.  IND is used as the risk-free rate: excess return per tick =
strategy return − IND return.

Usage
-----
    import pandas as pd
    from simulator import simulate

    df = pd.read_csv("data.csv")
    result = simulate(
        alphas      = my_signals,      # array-like, length N
        prices      = df["BBB"].values,
        ind_prices  = df["IND"].values,
    )
    print(result["sharpe"])
    result["cumulative_pnl"].plot()
"""

import numpy as np
import pandas as pd


def simulate(
    alphas,
    prices,
    ind_prices,
    capital: float = 10_000.0,
    transaction_cost: float = 0.0,
    ticks_per_year: int = None,
):
    """
    Simulate a strategy and return PnL statistics.

    Parameters
    ----------
    alphas : array-like of float, shape (N,)
        Signal values in [-1, 1].  Alpha at index t is the position held
        *from* tick t *until* tick t+1.  The last alpha is never traded
        (there is no t+1 price for it), so len(alphas) can equal len(prices).
    prices : array-like of float, shape (N,)
        Stock prices, one per tick.
    ind_prices : array-like of float, shape (N,)
        IND (risk-free) prices, one per tick.
    capital : float
        Notional capital in dollars.  A fully long position buys `capital`
        dollars worth of stock.
    transaction_cost : float
        Cost in dollars per unit of *notional* position change.  E.g. 0.001
        means 0.1% of the trade value is paid as a friction cost.
    ticks_per_year : int, optional
        Used to annualise the Sharpe ratio.  If None, defaults to the number
        of ticks in the data (i.e. one "year" = the full sample), which gives
        a non-annualised Sharpe.  Pass e.g. 252*390 for 1-minute US equity
        data or the actual ticks-per-year for your simulation.

    Returns
    -------
    dict with keys:
        pnl          : pd.Series  — gross P&L per tick (dollars)
        net_pnl      : pd.Series  — P&L after transaction costs per tick
        cumulative_pnl : pd.Series — cumulative net P&L
        excess_return  : pd.Series — strategy return minus IND return per tick
        total_pnl    : float  — total net P&L over the period
        mean_return  : float  — mean per-tick net strategy return
        std_return   : float  — std dev of per-tick net strategy return
        sharpe       : float  — annualised Sharpe ratio
        win_rate     : float  — fraction of ticks with positive net P&L
        max_drawdown : float  — maximum peak-to-trough drawdown in dollars
    """
    alphas     = np.asarray(alphas,     dtype=float)
    prices     = np.asarray(prices,     dtype=float)
    ind_prices = np.asarray(ind_prices, dtype=float)

    if len(alphas) != len(prices) or len(prices) != len(ind_prices):
        raise ValueError(
            f"alphas ({len(alphas)}), prices ({len(prices)}), and "
            f"ind_prices ({len(ind_prices)}) must all have the same length."
        )
    if np.any(alphas < -1) or np.any(alphas > 1):
        raise ValueError("All alpha values must be in [-1, 1].")

    N = len(prices)
    if N < 2:
        raise ValueError("Need at least 2 ticks to simulate.")

    # ── Per-tick price returns ────────────────────────────────────────────────
    # Return from tick t to t+1; defined for t = 0 … N-2
    price_ret = np.diff(prices)     / prices[:-1]   # fractional
    ind_ret   = np.diff(ind_prices) / ind_prices[:-1]

    # ── Positions and gross P&L ───────────────────────────────────────────────
    # Alpha at tick t → dollar position held from t to t+1
    positions = alphas[:-1] * capital             # dollar exposure per tick
    gross_pnl = positions * price_ret             # dollar P&L per tick

    # ── Transaction costs ─────────────────────────────────────────────────────
    # Cost is proportional to the *change* in notional position
    position_delta = np.diff(np.concatenate([[0.0], positions]))
    costs          = np.abs(position_delta) * transaction_cost
    net_pnl        = gross_pnl - costs

    # ── Returns (fractional, relative to capital) ────────────────────────────
    strategy_ret = net_pnl / capital              # net strategy return per tick
    excess_ret   = strategy_ret - ind_ret         # excess over risk-free

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    sigma = excess_ret.std(ddof=1)
    if ticks_per_year is None:
        ticks_per_year = N                        # non-annualised by default
    annualisation = np.sqrt(ticks_per_year)
    sharpe = (excess_ret.mean() / sigma * annualisation) if sigma > 0 else np.nan

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cum = np.cumsum(net_pnl)
    running_max  = np.maximum.accumulate(cum)
    drawdown     = cum - running_max
    max_drawdown = float(drawdown.min())

    # ── Package results ───────────────────────────────────────────────────────
    idx = np.arange(N - 1)
    return {
        "pnl"            : pd.Series(gross_pnl,  index=idx, name="pnl"),
        "net_pnl"        : pd.Series(net_pnl,    index=idx, name="net_pnl"),
        "cumulative_pnl" : pd.Series(cum,         index=idx, name="cumulative_pnl"),
        "excess_return"  : pd.Series(excess_ret,  index=idx, name="excess_return"),
        "total_pnl"      : float(cum[-1]),
        "mean_return"    : float(strategy_ret.mean()),
        "std_return"     : float(sigma),
        "sharpe"         : float(sharpe),
        "win_rate"       : float((net_pnl > 0).mean()),
        "max_drawdown"   : max_drawdown,
    }


def print_summary(result: dict) -> None:
    """Pretty-print the scalar statistics from a simulate() result."""
    print(f"  Total P&L      : ${result['total_pnl']:>12,.2f}")
    print(f"  Sharpe ratio   : {result['sharpe']:>12.4f}")
    print(f"  Win rate       : {result['win_rate']*100:>11.1f}%")
    print(f"  Max drawdown   : ${result['max_drawdown']:>12,.2f}")
    print(f"  Mean return/tk : {result['mean_return']:>12.6f}")
    print(f"  Std  return/tk : {result['std_return']:>12.6f}")

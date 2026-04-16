"""
Live Pair Trader
=================
Combines daily_pair_regression + threshold_optimizer and trades the result
live on the Rotman server.

Steps
-----
1. Runs daily_pair_regression on the CSV data and picks the Nth best pair.
2. Grid-searches (buy_in, back) thresholds to maximise Sharpe on that pair.
3. Connects to the Rotman server and trades using the spread strategy.

Usage
-----
    python live_pair_trader.py
"""

import os
import sys
import time
import logging
import requests
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROTMAN = os.path.join(_HERE, "../Rotman")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROTMAN)

from daily_pair_regression import run as get_daily_pairs
from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from settings import settings
from sklearn.linear_model import LinearRegression

# ── Config ────────────────────────────────────────────────────────────────────
PAIR_RANK      = 1      # 1 = best pair by avg R², 2 = second best, etc.
PAIR_RANK      = 1          # 1 = best pair by avg R², 2 = second best, etc.
CAPITAL        = 1_000_000
TRADE_FRACTION = 0.25

LOOKBACK       = 40         # ticks used to refit model and calibrate thresholds
REFIT_EVERY    = 10         # refit every N ticks
RISK_LIMIT     = 300        # stop trading if total P&L drops below −RISK_LIMIT

LOOP_INTERVAL = settings.get("loop_interval", 1)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _calibrate_thresholds(spread_arr, sd, grid_steps=25):
    """
    Grid-search (entry, exit) thresholds directly on a live spread array.
    Tries multipliers in [0.1, 4.0] × SD, both long and short sides.
    Returns (entry_thresh, exit_thresh).
    """
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

            in_pos    = 0   # 0=flat, +1=long spread, -1=short spread
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

    log.info(f"Live calibration: entry={best_entry:.4f}  exit={best_exit:.4f}  "
             f"Sharpe={best_sharpe:.4f}  (trades={len(trades) if 'trades' in dir() else 0})")
    return best_entry, best_exit


def place_market(client, ticker, action, quantity):
    if quantity <= 0:
        return
    try:
        result = client.place_order(ticker, OrderType.MARKET, quantity, action)
        log.info(f"  ORDER  {action.value:4s}  {quantity:>6d}  {ticker}  →  id={result.get('order_id')}")
    except Exception as e:
        log.error(f"  ORDER FAILED  {action.value} {quantity} {ticker}: {e}")


def run():
    # ── Step 1: pick pair from daily regression ───────────────────────────────
    log.info(f"Running daily pair regression — picking rank #{PAIR_RANK}...")
    _, _, _, pairs = get_daily_pairs()

    if PAIR_RANK > len(pairs):
        raise ValueError(f"PAIR_RANK={PAIR_RANK} but only {len(pairs)} pairs available.")

    chosen    = pairs[PAIR_RANK - 1]
    security1 = chosen["s1"]
    security2 = chosen["s2"]
    coef      = chosen["coef"]
    intercept = chosen["intercept"]

    log.info(f"Selected pair #{PAIR_RANK}: {security1}/{security2}  "
             f"avg R²={chosen['r2']:.4f}  coef={coef:.4f}  intercept={intercept:.4f}")

    # ── Step 2: connect and wait for market to open ───────────────────────────
    client = RotmanInteractiveTraderApi(
        api_key=settings["api_key"],
        api_host=settings["api_host"],
    )

    log.info("Waiting for market to open...")
    while True:
        try:
            if client.is_market_open():
                break
            log.info("Market not open yet, retrying...")
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            log.warning("Connection error, retrying...")
        time.sleep(1)
    log.info("Market is open.")

    # ── Flatten all existing positions ────────────────────────────────────────
    portfolio = client.get_portfolio()
    any_flat = False
    for ticker, sec in portfolio.items():
        pos = int(sec["position"])
        if pos > 0:
            log.info(f"Flattening existing LONG  {pos:>6d} {ticker}")
            place_market(client, ticker, OrderAction.SELL, pos)
            any_flat = True
        elif pos < 0:
            log.info(f"Flattening existing SHORT {abs(pos):>6d} {ticker}")
            place_market(client, ticker, OrderAction.BUY, abs(pos))
            any_flat = True
    if not any_flat:
        log.info("No existing positions to flatten.")

    # ── Step 3: refit params from live history + calibrate thresholds ─────────
    log.info("Fetching live history to refit spread parameters and calibrate thresholds...")
    hist1 = np.array([e["close"] for e in client.get_history(security1)])[-40:]
    hist2 = np.array([e["close"] for e in client.get_history(security2)])[-40:]

    model     = LinearRegression(fit_intercept=True).fit(hist1.reshape(-1, 1), hist2)
    coef      = float(model.coef_[0])
    intercept = float(model.intercept_)
    residuals = hist2 - (coef * hist1 + intercept)
    sd        = float(residuals.std())

    log.info(f"Live fit:  {security2} = {coef:.4f}·{security1} + {intercept:.4f}  SD={sd:.4f}")

    entry_thresh, exit_thresh = _calibrate_thresholds(residuals, sd)
    log.info(f"Entry at spread ≥ ±{entry_thresh:.4f}  |  Exit at ≤ ±{exit_thresh:.4f}")

    # ── Step 4: trade ─────────────────────────────────────────────────────────
    in_position = False
    tot_sec2    = 0
    tot_sec1    = 0
    pnl_history = []   # mark-to-market value each tick
    in_position  = False
    tot_sec2     = 0
    tot_sec1     = 0
    entry_price1 = 0.0
    entry_price2 = 0.0
    session_pnl  = 0.0     # cumulative realized P&L across closed trades
    pnl_history  = []      # MTM tick series for Sharpe
    price1_buf   = list(hist1)
    price2_buf   = list(hist2)
    tick_count   = 0

    log.info(f"Starting loop — {security2}/{security1}  "
             f"entry={entry_thresh:.4f}  exit={exit_thresh:.4f}"
             f"entry={entry_thresh:.4f}  exit={exit_thresh:.4f}  "
             f"risk_limit={RISK_LIMIT:,.0f}")

    def _realize_pnl(p1, p2):
        """Compute P&L of the current open position at prices p1/p2."""
        if tot_sec2 < 0:
            return abs(tot_sec2) * (entry_price2 - p2) + abs(tot_sec1) * (p1 - entry_price1)
        elif tot_sec2 > 0:
            return abs(tot_sec2) * (p2 - entry_price2) + abs(tot_sec1) * (entry_price1 - p1)
        return 0.0

    while True:
        try:
            if not client.is_market_open():
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            log.warning("Connection error checking market status, continuing...")
            time.sleep(LOOP_INTERVAL)
            continue

        try:
            portfolio = client.get_portfolio()
            price1    = portfolio[security1]["last"]
            price2    = portfolio[security2]["last"]
            spread    = price2 - (coef * price1 + intercept)
            portfolio  = client.get_portfolio()
            price1     = portfolio[security1]["last"]
            price2     = portfolio[security2]["last"]
            spread     = price2 - (coef * price1 + intercept)
            tick_count += 1

            price1_buf.append(price1)
            price2_buf.append(price2)

            # ── Sharpe ────────────────────────────────────────────────────────
            mtm = tot_sec1 * price1 + tot_sec2 * price2
            pnl_history.append(mtm)
            if len(pnl_history) >= 2:
                deltas = np.diff(pnl_history)
                sigma  = deltas.std()
                sharpe = deltas.mean() / sigma if sigma > 0 else 0.0
            else:
                sharpe = 0.0

            # ── Total P&L (realized + unrealized) ─────────────────────────────
            unrealized = _realize_pnl(price1, price2)
            total_pnl  = session_pnl + unrealized

            log.info(f"{security1}={price1:.4f}  {security2}={price2:.4f}  "
                     f"spread={spread:+.4f}  "
                     f"(entry≥{entry_thresh:.4f}  exit≤{exit_thresh:.4f})  "
                     f"Sharpe={sharpe:+.4f}"
                     f"(entry≥±{entry_thresh:.4f}  exit≤±{exit_thresh:.4f})  "
                     f"PnL={total_pnl:+,.0f}  Sharpe={sharpe:+.4f}")

            # ── Risk check ────────────────────────────────────────────────────
            if total_pnl < -RISK_LIMIT:
                log.warning(f"RISK LIMIT HIT — P&L {total_pnl:+,.0f} < -{RISK_LIMIT:,.0f}. "
                            f"Flattening all positions.")
                if in_position:
                    place_market(client, security2, OrderAction.BUY  if tot_sec2 < 0 else OrderAction.SELL, abs(tot_sec2))
                    place_market(client, security1, OrderAction.SELL if tot_sec1 > 0 else OrderAction.BUY,  abs(tot_sec1))
                break

            # ── Periodic refit ────────────────────────────────────────────────
            if tick_count % REFIT_EVERY == 0 and len(price1_buf) >= LOOKBACK:
                h1  = np.array(price1_buf[-LOOKBACK:])
                h2  = np.array(price2_buf[-LOOKBACK:])
                mdl = LinearRegression(fit_intercept=True).fit(h1.reshape(-1, 1), h2)
                coef      = float(mdl.coef_[0])
                intercept = float(mdl.intercept_)
                res       = h2 - (coef * h1 + intercept)
                sd        = float(res.std())
                entry_thresh, exit_thresh = _calibrate_thresholds(res, sd)
                spread = price2 - (coef * price1 + intercept)   # recalc with new params
                log.info(f"  REFIT  coef={coef:.4f}  intercept={intercept:.4f}  SD={sd:.4f}  "
                         f"entry=±{entry_thresh:.4f}  exit=±{exit_thresh:.4f}")

            # ── Trading logic ─────────────────────────────────────────────────
            notional = CAPITAL * TRADE_FRACTION
            qty2     = int(notional // price2)
            qty1     = int(notional // price1)

            if not in_position:
                if spread >= entry_thresh:
                    # Spread too high — short security2, long security1
                    log.info(f"ENTRY SHORT — sell {security2} ({qty2}), buy {security1} ({qty1})")
                    place_market(client, security2, OrderAction.SELL, qty2)
                    place_market(client, security1, OrderAction.BUY,  qty1)
                    tot_sec2    = -qty2
                    tot_sec1    =  qty1
                    tot_sec2 = -qty2; tot_sec1 = qty1
                    entry_price1 = price1; entry_price2 = price2
                    in_position = True

                elif spread <= -entry_thresh:
                    # Spread too low — long security2, short security1
                    log.info(f"ENTRY LONG  — buy {security2} ({qty2}), sell {security1} ({qty1})")
                    place_market(client, security2, OrderAction.BUY,  qty2)
                    place_market(client, security1, OrderAction.SELL, qty1)
                    tot_sec2    =  qty2
                    tot_sec1    = -qty1
                    tot_sec2 = qty2; tot_sec1 = -qty1
                    entry_price1 = price1; entry_price2 = price2
                    in_position = True

            elif in_position:
                if tot_sec2 < 0:
                    # Currently SHORT spread (short sec2, long sec1)
                    if spread <= exit_thresh:
                        log.info("EXIT SHORT — closing position")
                    # SHORT spread — flip checked first (more extreme than exit)
                    if spread <= -entry_thresh:
                        log.info(f"FLIP SHORT→LONG — {security2} ({abs(tot_sec2)}→{qty2})")
                        session_pnl += _realize_pnl(price1, price2)
                        place_market(client, security2, OrderAction.BUY,  abs(tot_sec2))
                        place_market(client, security1, OrderAction.SELL, abs(tot_sec1))
                        tot_sec2 = 0; tot_sec1 = 0; in_position = False
                        place_market(client, security2, OrderAction.BUY,  qty2)
                        place_market(client, security1, OrderAction.SELL, qty1)
                        tot_sec2 = qty2; tot_sec1 = -qty1
                        entry_price1 = price1; entry_price2 = price2

                    elif spread <= -entry_thresh:
                        # Spread flipped to the other extreme — reverse directly
                        log.info(f"FLIP SHORT→LONG — close short, open long  {security2} ({abs(tot_sec2)}→{qty2})")
                    elif spread <= exit_thresh:
                        log.info("EXIT SHORT — closing position")
                        session_pnl += _realize_pnl(price1, price2)
                        place_market(client, security2, OrderAction.BUY,  abs(tot_sec2))
                        place_market(client, security1, OrderAction.SELL, abs(tot_sec1))
                        place_market(client, security2, OrderAction.BUY,  qty2)
                        place_market(client, security1, OrderAction.SELL, qty1)
                        tot_sec2 =  qty2; tot_sec1 = -qty1
                        tot_sec2 = 0; tot_sec1 = 0; in_position = False

                elif tot_sec2 > 0:
                    # Currently LONG spread (long sec2, short sec1)
                    if spread >= -exit_thresh:
                        log.info("EXIT LONG  — closing position")
                    # LONG spread — flip checked first (more extreme than exit)
                    if spread >= entry_thresh:
                        log.info(f"FLIP LONG→SHORT — {security2} ({abs(tot_sec2)}→{qty2})")
                        session_pnl += _realize_pnl(price1, price2)
                        place_market(client, security2, OrderAction.SELL, abs(tot_sec2))
                        place_market(client, security1, OrderAction.BUY,  abs(tot_sec1))
                        tot_sec2 = 0; tot_sec1 = 0; in_position = False
                        place_market(client, security2, OrderAction.SELL, qty2)
                        place_market(client, security1, OrderAction.BUY,  qty1)
                        tot_sec2 = -qty2; tot_sec1 = qty1
                        entry_price1 = price1; entry_price2 = price2

                    elif spread >= entry_thresh:
                        # Spread flipped to the other extreme — reverse directly
                        log.info(f"FLIP LONG→SHORT — close long, open short  {security2} ({abs(tot_sec2)}→{qty2})")
                    elif spread >= -exit_thresh:
                        log.info("EXIT LONG  — closing position")
                        session_pnl += _realize_pnl(price1, price2)
                        place_market(client, security2, OrderAction.SELL, abs(tot_sec2))
                        place_market(client, security1, OrderAction.BUY,  abs(tot_sec1))
                        place_market(client, security2, OrderAction.SELL, qty2)
                        place_market(client, security1, OrderAction.BUY,  qty1)
                        tot_sec2 = -qty2; tot_sec1 =  qty1
                        tot_sec2 = 0; tot_sec1 = 0; in_position = False

        except Exception as e:
            log.error(f"Tick error: {e}")

        time.sleep(LOOP_INTERVAL)

    # ── Market closed ─────────────────────────────────────────────────────────
    log.info(f"Session ended — realized P&L: {session_pnl:+,.0f}")
    if in_position:
        log.info("Market closed — closing open position.")
        log.info("Closing open position.")
        portfolio = client.get_portfolio()
        price1    = portfolio[security1]["last"]
        price2    = portfolio[security2]["last"]
        place_market(client, security2, OrderAction.BUY,  abs(tot_sec2))
        place_market(client, security1, OrderAction.SELL, abs(tot_sec1))
        place_market(client, security2, OrderAction.BUY  if tot_sec2 < 0 else OrderAction.SELL, abs(tot_sec2))
        place_market(client, security1, OrderAction.SELL if tot_sec1 > 0 else OrderAction.BUY,  abs(tot_sec1))

    log.info("Done.")


if __name__ == "__main__":
    run()
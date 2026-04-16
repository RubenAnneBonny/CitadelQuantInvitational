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
CAPITAL        = 1_000_000
TRADE_FRACTION = 0.25

LOOKBACK     = 40       # (Y) ticks used for each parameter fit
REFIT_EVERY  = 10       # (X) refit every N ticks when flat

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
    any_flat  = False
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

    # ── Step 3: initial fit from live history ────────────────────────────────
    log.info(f"Fetching live history (last {LOOKBACK} ticks) to fit parameters...")
    hist1 = np.array([e["close"] for e in client.get_history(security1)])[-LOOKBACK:]
    hist2 = np.array([e["close"] for e in client.get_history(security2)])[-LOOKBACK:]

    def _refit(h1, h2):
        mdl       = LinearRegression(fit_intercept=True).fit(h1.reshape(-1, 1), h2)
        c         = float(mdl.coef_[0])
        i         = float(mdl.intercept_)
        res       = h2 - (c * h1 + i)
        s         = float(res.std())
        et, xt    = _calibrate_thresholds(res, s)
        log.info(f"  FIT  {security2} = {c:.4f}·{security1} + {i:.4f}  SD={s:.4f}  "
                 f"entry=±{et:.4f}  exit=±{xt:.4f}")
        return c, i, et, xt

    coef, intercept, entry_thresh, exit_thresh = _refit(hist1, hist2)

    # ── Step 4: trade ─────────────────────────────────────────────────────────
    in_position = False
    tot_sec2    = 0
    tot_sec1    = 0
    price1_buf  = list(hist1)
    price2_buf  = list(hist2)
    tick_count  = 0

    log.info(f"Starting loop — {security2}/{security1}  "
             f"entry={entry_thresh:.4f}  exit={exit_thresh:.4f}  "
             f"refit every {REFIT_EVERY} ticks (lookback {LOOKBACK})")

    while True:
        try:
            if not client.is_market_open():
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            log.warning("Connection error checking market status, continuing...")
            time.sleep(LOOP_INTERVAL)
            continue

        try:
            portfolio  = client.get_portfolio()
            price1     = portfolio[security1]["last"]
            price2     = portfolio[security2]["last"]
            tick_count += 1

            price1_buf.append(price1)
            price2_buf.append(price2)

            # ── Refit when flat every REFIT_EVERY ticks ───────────────────────
            if not in_position and tick_count % REFIT_EVERY == 0:
                h1 = np.array(price1_buf[-LOOKBACK:])
                h2 = np.array(price2_buf[-LOOKBACK:])
                coef, intercept, entry_thresh, exit_thresh = _refit(h1, h2)

            spread = price2 - (coef * price1 + intercept)

            log.info(f"{security1}={price1:.4f}  {security2}={price2:.4f}  "
                     f"spread={spread:+.4f}  (entry≥{entry_thresh:.4f}  exit≤{exit_thresh:.4f})")

            if not in_position:
                notional = CAPITAL * TRADE_FRACTION
                qty2     = int(notional // price2)
                qty1     = int(notional // price1)

                if spread >= entry_thresh:
                    # Spread too high — short security2, long security1
                    log.info(f"ENTRY SHORT — sell {security2} ({qty2}), buy {security1} ({qty1})")
                    place_market(client, security2, OrderAction.SELL, qty2)
                    place_market(client, security1, OrderAction.BUY,  qty1)
                    tot_sec2    = -qty2
                    tot_sec1    =  qty1
                    in_position = True

                elif spread <= -entry_thresh:
                    # Spread too low — long security2, short security1
                    log.info(f"ENTRY LONG  — buy {security2} ({qty2}), sell {security1} ({qty1})")
                    place_market(client, security2, OrderAction.BUY,  qty2)
                    place_market(client, security1, OrderAction.SELL, qty1)
                    tot_sec2    =  qty2
                    tot_sec1    = -qty1
                    in_position = True

            elif in_position:
                # Exit when spread reverts back toward zero past exit threshold
                if tot_sec2 < 0 and spread <= exit_thresh:
                    log.info("EXIT SHORT — closing position")
                    place_market(client, security2, OrderAction.BUY,  abs(tot_sec2))
                    place_market(client, security1, OrderAction.SELL, abs(tot_sec1))
                    tot_sec2 = 0; tot_sec1 = 0; in_position = False

                elif tot_sec2 > 0 and spread >= -exit_thresh:
                    log.info("EXIT LONG  — closing position")
                    place_market(client, security2, OrderAction.SELL, abs(tot_sec2))
                    place_market(client, security1, OrderAction.BUY,  abs(tot_sec1))
                    tot_sec2 = 0; tot_sec1 = 0; in_position = False

        except Exception as e:
            log.error(f"Tick error: {e}")

        time.sleep(LOOP_INTERVAL)

    # ── Market closed ─────────────────────────────────────────────────────────
    if in_position:
        log.info("Market closed — closing open position.")
        portfolio = client.get_portfolio()
        price1    = portfolio[security1]["last"]
        price2    = portfolio[security2]["last"]
        place_market(client, security2, OrderAction.BUY,  abs(tot_sec2))
        place_market(client, security1, OrderAction.SELL, abs(tot_sec1))

    log.info("Done.")


if __name__ == "__main__":
    run()

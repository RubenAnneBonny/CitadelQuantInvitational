"""
Live Pair Trader — Percentage / Ratio Model
============================================
Same as live_pair_trader.py but uses the ratio spread:

    spread = price2 / price1 − ratio

instead of the absolute spread  price2 − (coef·price1 + intercept).

Usage
-----
    python live_pair_trader_pct.py
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROTMAN = os.path.join(_HERE, "../Rotman")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROTMAN)

from daily_pair_regression_pct  import run as get_daily_pairs
from threshold_optimizer_pct    import find_best_thresholds_pct
from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from settings import settings

# ── Config ────────────────────────────────────────────────────────────────────
PAIR_RANK      = 1      # 1 = best pair by avg R², 2 = second best, etc.
TRAIN_DAYS     = 30
CAPITAL        = 1_000_000
TRADE_FRACTION = 0.25

DATA_FILE     = os.path.join(_HERE, "../Competition/Alpha Testing V2/data.csv")
LOOP_INTERVAL = settings.get("loop_interval", 1)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def place_market(client, ticker, action, quantity):
    if quantity <= 0:
        return
    try:
        result = client.place_order(ticker, OrderType.MARKET, quantity, action)
        log.info(f"  ORDER  {action.value:4s}  {quantity:>6d}  {ticker}  →  id={result.get('order_id')}")
    except Exception as e:
        log.error(f"  ORDER FAILED  {action.value} {quantity} {ticker}: {e}")


def run():
    # ── Step 1: pick pair from daily ratio regression ─────────────────────────
    log.info(f"Running daily ratio regression — picking rank #{PAIR_RANK}...")
    _, _, pairs = get_daily_pairs()

    if PAIR_RANK > len(pairs):
        raise ValueError(f"PAIR_RANK={PAIR_RANK} but only {len(pairs)} pairs available.")

    chosen    = pairs[PAIR_RANK - 1]
    security1 = chosen["s1"]
    security2 = chosen["s2"]
    ratio     = chosen["ratio"]

    log.info(f"Selected pair #{PAIR_RANK}: {security1}/{security2}  "
             f"avg R²={chosen['r2']:.4f}  ratio={ratio:.6f}")

    # ── Step 2: find best thresholds ──────────────────────────────────────────
    log.info("Optimising thresholds...")
    df = pd.read_csv(DATA_FILE)
    buy_in, back, sd, best_sharpe = find_best_thresholds_pct(
        df, security1, security2, ratio, train_days=TRAIN_DAYS
    )

    entry_thresh = buy_in * sd
    exit_thresh  = back   * sd

    log.info(f"Best thresholds:  buy_in={buy_in:.4f}  back={back:.4f}  "
             f"SD={sd:.6f}  Sharpe={best_sharpe:.4f}")
    log.info(f"Entry at ratio spread ≥ ±{entry_thresh:.6f}  |  Exit at ≤ ±{exit_thresh:.6f}")

    # ── Step 3: connect to Rotman ─────────────────────────────────────────────
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

    # ── Step 4: trade ─────────────────────────────────────────────────────────
    in_position = False
    tot_sec2    = 0
    tot_sec1    = 0

    log.info(f"Starting loop — spread = {security2}/{security1} − {ratio:.6f}")

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
            spread    = price2 / price1 - ratio

            log.info(f"{security1}={price1:.4f}  {security2}={price2:.4f}  "
                     f"ratio={price2/price1:.6f}  spread={spread:+.6f}  "
                     f"(entry≥±{entry_thresh:.6f}  exit≤±{exit_thresh:.6f})")

            if not in_position:
                notional = CAPITAL * TRADE_FRACTION
                qty2     = int(notional // price2)
                qty1     = int(notional // price1)

                if spread >= entry_thresh:
                    # Ratio too high — security2 is expensive relative to security1
                    log.info(f"ENTRY SHORT — sell {security2} ({qty2}), buy {security1} ({qty1})")
                    place_market(client, security2, OrderAction.SELL, qty2)
                    place_market(client, security1, OrderAction.BUY,  qty1)
                    tot_sec2    = -qty2
                    tot_sec1    =  qty1
                    in_position = True

                elif spread <= -entry_thresh:
                    # Ratio too low — security2 is cheap relative to security1
                    log.info(f"ENTRY LONG  — buy {security2} ({qty2}), sell {security1} ({qty1})")
                    place_market(client, security2, OrderAction.BUY,  qty2)
                    place_market(client, security1, OrderAction.SELL, qty1)
                    tot_sec2    =  qty2
                    tot_sec1    = -qty1
                    in_position = True

            elif in_position:
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

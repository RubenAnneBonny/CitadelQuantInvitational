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
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROTMAN = os.path.join(_HERE, "../Rotman")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROTMAN)

from daily_pair_regression import run as get_daily_pairs
from threshold_optimizer   import find_best_thresholds
from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from settings import settings

# ── Config ────────────────────────────────────────────────────────────────────
PAIR_RANK  = 1      # 1 = best pair by avg R², 2 = second best, etc.
TRAIN_DAYS = 30     # days used to fit thresholds (must match your CSV training window)
CAPITAL    = 1_000_000
TRADE_FRACTION = 0.25

DATA_FILE  = os.path.join(_HERE, "../Competition/Alpha Testing V2/data.csv")
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

    # ── Step 2: find best thresholds ──────────────────────────────────────────
    log.info("Optimising thresholds...")
    df = pd.read_csv(DATA_FILE)
    buy_in, back, sd, best_sharpe = find_best_thresholds(
        df, security1, security2, coef, intercept, train_days=TRAIN_DAYS
    )

    entry_thresh = buy_in * sd
    exit_thresh  = back   * sd

    log.info(f"Best thresholds:  buy_in={buy_in:.4f}  back={back:.4f}  "
             f"SD={sd:.4f}  Sharpe={best_sharpe:.4f}")
    log.info(f"Entry at spread ≥ {entry_thresh:.4f}  |  Exit at spread ≤ {exit_thresh:.4f}")

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

    log.info(f"Starting loop — {security2}/{security1}  "
             f"entry={entry_thresh:.4f}  exit={exit_thresh:.4f}")

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

            log.info(f"{security1}={price1:.4f}  {security2}={price2:.4f}  "
                     f"spread={spread:+.4f}  (entry≥{entry_thresh:.4f}  exit≤{exit_thresh:.4f})")

            if spread >= entry_thresh and not in_position:
                notional = CAPITAL * TRADE_FRACTION
                qty2     = int(notional // price2)
                qty1     = int(notional // price1)

                log.info(f"ENTRY — short {security2} ({qty2}), long {security1} ({qty1})")
                place_market(client, security2, OrderAction.SELL, qty2)
                place_market(client, security1, OrderAction.BUY,  qty1)

                tot_sec2    = -qty2
                tot_sec1    =  qty1
                in_position = True

            elif spread <= exit_thresh and in_position:
                log.info("EXIT — closing position")
                place_market(client, security2, OrderAction.BUY,  abs(tot_sec2))
                place_market(client, security1, OrderAction.SELL, abs(tot_sec1))

                tot_sec2    = 0
                tot_sec1    = 0
                in_position = False

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
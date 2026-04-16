"""
Pair Trader — Live Rotman
=========================
Replicates the Backtest.ipynb mean-reversion strategy on the Rotman server.

Strategy
--------
  spread = security2_price - (coef * security1_price + intercept)
  Entry : spread > buy_in * sd  →  short security2, long security1
  Exit  : spread < back * sd    →  close both legs

Usage
-----
    python pair_trader.py
"""

import time
import logging
import requests

from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
SECURITY1  = "DDD"                # independent variable (hedge leg)
SECURITY2  = "BBB"                # dependent variable   (primary leg)

INTERCEPT  = -0.878753  # from OLS fit
COEF       =   0.508236         # from OLS fit
SD         =   5.792330 # spread std dev from training data

BUY_IN     = 1.2222               # entry threshold (multiples of SD)
BACK       = 0.1                  # exit threshold  (multiples of SD)

CAPITAL        = 1_000_000
TRADE_FRACTION = 0.25             # fraction of capital per leg

LOOP_INTERVAL = settings.get("loop_interval", 1)  # seconds between ticks
# ─────────────────────────────────────────────────────────────────────────────


def get_current_prices(client: RotmanInteractiveTraderApi) -> tuple[float, float]:
    """Return (security1_last, security2_last) from the live portfolio."""
    portfolio = client.get_portfolio()
    return portfolio[SECURITY1]["last"], portfolio[SECURITY2]["last"]


def compute_spread(price1: float, price2: float, intercept: float, coef: float) -> float:
    return price2 - (coef * price1 + intercept)


def place_market(client: RotmanInteractiveTraderApi,
                 ticker: str, action: OrderAction, quantity: int) -> None:
    """Place a market order and log the result."""
    if quantity <= 0:
        return
    try:
        result = client.place_order(ticker, OrderType.MARKET, quantity, action)
        log.info(f"  ORDER  {action.value:4s}  {quantity:>6d}  {ticker}  →  id={result.get('order_id')}")
    except Exception as e:
        log.error(f"  ORDER FAILED  {action.value} {quantity} {ticker}: {e}")


def run() -> None:
    client = RotmanInteractiveTraderApi(
        api_key=settings["api_key"],
        api_host=settings["api_host"],
    )

    # ── Wait for market to open ───────────────────────────────────────────────
    log.info("Waiting for market to open...")
    while True:
        try:
            if client.is_market_open():
                break
            log.info("Market not open yet, retrying...")
        except requests.exceptions.ConnectionError:
            log.warning("Connection error while checking market status, retrying...")
        except requests.exceptions.ConnectTimeout:
            log.warning("Connection timed out while checking market status, retrying...")
        time.sleep(1)
    log.info("Market is open.")

    # ── Log parameters ────────────────────────────────────────────────────────
    log.info(f"Pair:  {SECURITY2} = {COEF:.4f} * {SECURITY1} + {INTERCEPT:.4f}")
    log.info(f"SD: {SD:.4f}  |  Entry: {BUY_IN}×SD = {BUY_IN*SD:.4f}  |  Exit: {BACK}×SD = {BACK*SD:.4f}")

    # ── State ─────────────────────────────────────────────────────────────────
    in_position = False     # True when a trade pair is open
    tot_sec2    = 0         # shares of security2 (negative = short)
    tot_sec1    = 0         # shares of security1 (positive = long hedge)

    entry_thresh = BUY_IN * SD
    exit_thresh  = BACK   * SD

    # ── Main loop ─────────────────────────────────────────────────────────────
    log.info("Starting trading loop...")
    prev_spread = None

    while True:
        try:
            if not client.is_market_open():
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            log.warning("Connection error checking market status, continuing...")
            time.sleep(LOOP_INTERVAL)
            continue
        try:
            price1, price2 = get_current_prices(client)
            spread = compute_spread(price1, price2, INTERCEPT, COEF)

            log.info(
                f"{SECURITY1}={price1:.4f}  {SECURITY2}={price2:.4f}  "
                f"spread={spread:+.4f}  (threshold ±{entry_thresh:.4f})"
            )

            if prev_spread is not None:

                # ── Entry: spread crossed above entry threshold ────────────────
                if spread >= entry_thresh and prev_spread < entry_thresh and not in_position:
                    notional  = CAPITAL * TRADE_FRACTION
                    qty_sec2  = int(notional // price2)
                    qty_sec1  = int(notional // price1)

                    log.info(f"ENTRY — short {SECURITY2}, long {SECURITY1}")
                    place_market(client, SECURITY2, OrderAction.SELL, qty_sec2)
                    place_market(client, SECURITY1, OrderAction.BUY,  qty_sec1)

                    tot_sec2    = -qty_sec2
                    tot_sec1    =  qty_sec1
                    in_position = True

                # ── Exit: spread fell back below exit threshold ────────────────
                elif spread <= exit_thresh and prev_spread > exit_thresh and in_position:
                    log.info(f"EXIT — closing position")
                    place_market(client, SECURITY2, OrderAction.BUY,  abs(tot_sec2))
                    place_market(client, SECURITY1, OrderAction.SELL, abs(tot_sec1))

                    tot_sec2    = 0
                    tot_sec1    = 0
                    in_position = False

            prev_spread = spread

        except Exception as e:
            log.error(f"Tick error: {e}")

        time.sleep(LOOP_INTERVAL)

    # ── Market closed — close any open position ───────────────────────────────
    if in_position:
        log.info("Market closed — closing open position.")
        price1, price2 = get_current_prices(client)
        place_market(client, SECURITY2, OrderAction.BUY,  abs(tot_sec2))
        place_market(client, SECURITY1, OrderAction.SELL, abs(tot_sec1))

    log.info("Done.")


if __name__ == "__main__":
    run()

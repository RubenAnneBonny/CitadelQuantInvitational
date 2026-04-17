"""
Together — Dual Strategy Runner
================================
Runs two pair-trading strategies simultaneously, each with half the capital.

  Strategy A  (fix_first_version)
      Dynamically selects the best pair from daily_pair_regression.
      OLS spread, periodic refit every REFIT_EVERY ticks.

  Strategy B  (Primitive / IND-ETF)
      Trades IND/ETF using the same OLS + calibrate approach.
      Periodic refit every REFIT_EVERY ticks.

Usage
-----
    python together.py
"""

import os
import sys
import time
import logging
import requests
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_DR     = os.path.join(_HERE, "../Daily Regression")
_ROTMAN = os.path.join(_HERE, "../Rotman")
sys.path.insert(0, _HERE)
sys.path.insert(0, _DR)
sys.path.insert(0, _ROTMAN)

from daily_pair_regression import run as get_daily_pairs
from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi, OrderType, OrderAction
from settings import settings
from sklearn.linear_model import LinearRegression

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_CAPITAL  = 20_000_000   # split evenly between the two strategies
HALF_CAPITAL   = TOTAL_CAPITAL // 2
TRADE_FRACTION = 0.4          # fraction of each half used per trade

LOOKBACK       = 40           # ticks used when fitting the OLS model
REFIT_EVERY    = 5            # refit every N ticks (regardless of position)
PAIR_RANK      = 1            # which pair from daily_pair_regression for Strategy A

# Strategy B always trades this pair
B_SEC1 = "IND"
B_SEC2 = "ETF"

BAD_TICKS = [391, 390, 389, 0, 1, 2]

LOOP_INTERVAL = settings.get("loop_interval", 1)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Shared helpers ─────────────────────────────────────────────────────────────

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


def place_market(client, ticker, action, quantity, label=""):
    if quantity <= 0:
        return
    try:
        result = client.place_order(ticker, OrderType.MARKET, quantity, action)
        log.info(f"  [{label}] ORDER  {action.value:4s}  {quantity:>6d}  {ticker}"
                 f"  →  id={result.get('order_id')}")
    except Exception as e:
        log.error(f"  [{label}] ORDER FAILED  {action.value} {quantity} {ticker}: {e}")


def _flatten_all(client):
    """Cancel all open orders then close every non-zero tradeable position."""
    try:
        client.cancel_all_orders()
        log.info("  Cancelled all open orders.")
    except Exception as e:
        log.error(f"  cancel_all_orders failed: {e}")

    for attempt in range(15):
        portfolio  = client.get_portfolio()
        still_open = False

        for ticker, sec in portfolio.items():
            if not sec.get("is_tradeable", False):
                continue
            pos = round(float(sec.get("position", 0)))
            if pos == 0:
                continue
            still_open = True
            action     = OrderAction.SELL if pos > 0 else OrderAction.BUY
            max_chunk  = int(sec.get("max_trade_size", abs(pos))) or abs(pos)
            chunk      = min(abs(pos), max_chunk)
            log.info(f"  [{attempt+1}] {ticker:>6s}  pos={pos:>8d}  → {action.value} {chunk}")
            place_market(client, ticker, action, chunk, label="FLATTEN")

        if not still_open:
            log.info("  All positions flat.")
            return True

        time.sleep(0.3)

    log.warning("  Could not fully flatten after 15 attempts.")
    return False


def _fit_model(buf1, buf2):
    """OLS fit + calibrate. Returns (coef, intercept, entry_thresh, exit_thresh)."""
    h1  = np.array(buf1[-LOOKBACK:])
    h2  = np.array(buf2[-LOOKBACK:])
    mdl = LinearRegression(fit_intercept=True).fit(h1.reshape(-1, 1), h2)
    c   = float(mdl.coef_[0])
    ic  = float(mdl.intercept_)
    res = h2 - (c * h1 + ic)
    sd  = float(res.std())
    en, ex = _calibrate_thresholds(res, sd)
    return c, ic, en, ex


# ── Strategy state container ───────────────────────────────────────────────────

class Strategy:
    def __init__(self, name, sec1, sec2, capital, trade_fraction):
        self.name    = name
        self.sec1    = sec1
        self.sec2    = sec2
        self.capital = capital
        self.tfrac   = trade_fraction

        self.coef         = 0.0
        self.intercept    = 0.0
        self.entry_thresh = 1.0
        self.exit_thresh  = 0.3

        self.in_position = False
        self.tot_sec1    = 0
        self.tot_sec2    = 0

        self.buf1 = []
        self.buf2 = []

    def seed(self, hist1, hist2):
        """Initial fit from startup history."""
        self.buf1 = list(hist1)
        self.buf2 = list(hist2)
        self.coef, self.intercept, self.entry_thresh, self.exit_thresh = \
            _fit_model(self.buf1, self.buf2)
        log.info(f"[{self.name}] Initial fit:  {self.sec2} = {self.coef:.4f}·{self.sec1}"
                 f" + {self.intercept:.4f}  entry=±{self.entry_thresh:.4f}"
                 f"  exit=±{self.exit_thresh:.4f}")

    def refit(self, price1, price2):
        """Refit model and recalculate spread."""
        if len(self.buf1) < LOOKBACK:
            return
        self.coef, self.intercept, self.entry_thresh, self.exit_thresh = \
            _fit_model(self.buf1, self.buf2)
        spread = price2 - (self.coef * price1 + self.intercept)
        log.info(f"  [{self.name}] REFIT  coef={self.coef:.4f}  intercept={self.intercept:.4f}"
                 f"  entry=±{self.entry_thresh:.4f}  exit=±{self.exit_thresh:.4f}"
                 f"  spread={spread:+.4f}")

    def tick(self, client, portfolio, tick_num):
        """Run one tick of strategy logic. Returns the spread."""
        price1 = portfolio[self.sec1]["last"]
        price2 = portfolio[self.sec2]["last"]

        self.buf1.append(price1)
        self.buf2.append(price2)

        # Periodic refit
        if tick_num % REFIT_EVERY == 0 and len(self.buf1) >= LOOKBACK:
            self.refit(price1, price2)

        spread = price2 - (self.coef * price1 + self.intercept)

        notional = self.capital * self.tfrac
        qty2     = int(notional // price2)
        qty1     = int(notional // price1)

        log.info(f"  [{self.name}]  {self.sec1}={price1:.4f}  {self.sec2}={price2:.4f}"
                 f"  spread={spread:+.4f}"
                 f"  (entry≥±{self.entry_thresh:.4f}  exit≤±{self.exit_thresh:.4f})"
                 f"  pos={self.tot_sec2:+d}/{self.tot_sec1:+d}")

        if not self.in_position:
            if spread >= self.entry_thresh:
                log.info(f"  [{self.name}] ENTRY SHORT — sell {self.sec2} ({qty2}),"
                         f" buy {self.sec1} ({qty1})")
                place_market(client, self.sec2, OrderAction.SELL, qty2, self.name)
                place_market(client, self.sec1, OrderAction.BUY,  qty1, self.name)
                self.tot_sec2 = -qty2; self.tot_sec1 = qty1
                self.in_position = True

            elif spread <= -self.entry_thresh:
                log.info(f"  [{self.name}] ENTRY LONG  — buy {self.sec2} ({qty2}),"
                         f" sell {self.sec1} ({qty1})")
                place_market(client, self.sec2, OrderAction.BUY,  qty2, self.name)
                place_market(client, self.sec1, OrderAction.SELL, qty1, self.name)
                self.tot_sec2 = qty2; self.tot_sec1 = -qty1
                self.in_position = True

        else:
            if self.tot_sec2 < 0 and spread <= self.exit_thresh:
                log.info(f"  [{self.name}] EXIT SHORT")
                place_market(client, self.sec2, OrderAction.BUY,  abs(self.tot_sec2), self.name)
                place_market(client, self.sec1, OrderAction.SELL, abs(self.tot_sec1), self.name)
                self.tot_sec2 = 0; self.tot_sec1 = 0; self.in_position = False

            elif self.tot_sec2 > 0 and spread >= -self.exit_thresh:
                log.info(f"  [{self.name}] EXIT LONG")
                place_market(client, self.sec2, OrderAction.SELL, abs(self.tot_sec2), self.name)
                place_market(client, self.sec1, OrderAction.BUY,  abs(self.tot_sec1), self.name)
                self.tot_sec2 = 0; self.tot_sec1 = 0; self.in_position = False

        return spread

    def close(self, client, portfolio):
        """Close any open position at market close."""
        if self.in_position:
            log.info(f"  [{self.name}] Closing open position at market close.")
            place_market(client, self.sec2,
                         OrderAction.BUY  if self.tot_sec2 < 0 else OrderAction.SELL,
                         abs(self.tot_sec2), self.name)
            place_market(client, self.sec1,
                         OrderAction.SELL if self.tot_sec1 > 0 else OrderAction.BUY,
                         abs(self.tot_sec1), self.name)

    def reset(self):
        """Reset position state (e.g. after bad tick flatten)."""
        self.in_position = False
        self.tot_sec1    = 0
        self.tot_sec2    = 0


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    # ── Strategy A: pick pair from daily regression ───────────────────────────
    log.info(f"Running daily pair regression — picking rank #{PAIR_RANK}...")
    _, _, _, pairs = get_daily_pairs()
    if PAIR_RANK > len(pairs):
        raise ValueError(f"PAIR_RANK={PAIR_RANK} but only {len(pairs)} pairs available.")
    chosen = pairs[PAIR_RANK - 1]
    a_sec1 = chosen["s1"]
    a_sec2 = chosen["s2"]
    log.info(f"[A] Selected pair: {a_sec1}/{a_sec2}  avg R²={chosen['r2']:.4f}")

    strat_a = Strategy("A", a_sec1, a_sec2, HALF_CAPITAL, TRADE_FRACTION)
    strat_b = Strategy("B", B_SEC1, B_SEC2, HALF_CAPITAL, TRADE_FRACTION)

    # ── Connect and wait for market ───────────────────────────────────────────
    client = RotmanInteractiveTraderApi(
        api_key=settings["api_key"],
        api_host=settings["api_host"],
    )

    log.info("Waiting for market to open...")
    while True:
        try:
            status = client.get_case()["status"]
            if status == "ACTIVE":
                break
            log.info(f"Market status: {status} — waiting...")
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            log.warning("Connection error, retrying...")
        time.sleep(1)
    log.info("Market is open.")

    # ── Flatten existing positions ────────────────────────────────────────────
    log.info("Flattening all existing positions...")
    _flatten_all(client)

    # ── Seed both strategies with live history ────────────────────────────────
    log.info("Fetching history for initial model fit...")
    hist_a1 = np.array([e["close"] for e in client.get_history(a_sec1)])[-LOOKBACK:]
    hist_a2 = np.array([e["close"] for e in client.get_history(a_sec2)])[-LOOKBACK:]
    hist_b1 = np.array([e["close"] for e in client.get_history(B_SEC1)])[-LOOKBACK:]
    hist_b2 = np.array([e["close"] for e in client.get_history(B_SEC2)])[-LOOKBACK:]

    strat_a.seed(hist_a1, hist_a2)
    strat_b.seed(hist_b1, hist_b2)

    # ── Main loop ─────────────────────────────────────────────────────────────
    tick_count = 0

    while True:
        try:
            status = client.get_case()["status"]
            if status == "STOPPED":
                break
            if status == "PAUSED":
                time.sleep(LOOP_INTERVAL)
                continue
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
            log.warning("Connection error, retrying...")
            time.sleep(LOOP_INTERVAL)
            continue

        try:
            curr      = client.get_case()
            portfolio = client.get_portfolio()
            tick_count += 1

            log.info(f"─── tick={curr['tick']} ───────────────────────────────")

            if curr["tick"] in BAD_TICKS:
                log.warning(f"BAD TICK {curr['tick']} — flattening all positions.")
                _flatten_all(client)
                strat_a.reset()
                strat_b.reset()
                continue

            strat_a.tick(client, portfolio, tick_count)
            strat_b.tick(client, portfolio, tick_count)

        except Exception as e:
            log.error(f"Tick error: {e}")

        time.sleep(LOOP_INTERVAL)

    # ── Market closed ─────────────────────────────────────────────────────────
    log.info("Market closed.")
    portfolio = client.get_portfolio()
    strat_a.close(client, portfolio)
    strat_b.close(client, portfolio)
    log.info("Done.")


if __name__ == "__main__":
    run()

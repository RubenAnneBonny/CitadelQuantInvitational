"""
RIT Quick Test Script
─────────────────────
Run this file to verify your connection to RIT, visualize live price data,
and place a small test order.

Requirements:
    pip install requests matplotlib

Make sure RIT is open and a case is running before executing.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Allow imports from the Competition folder one level up
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Competition"))
from rit_client import RITClient

# ── Settings ───────────────────────────────────────────────────────────────────
TICKER = "CRZY"

# Test order — keep quantity small
TEST_ORDER_TICKER   = "CRZY"
TEST_ORDER_QUANTITY = 10
TEST_ORDER_TYPE     = "LIMIT"   # "MARKET" or "LIMIT"
TEST_ORDER_SIDE     = "BUY"     # "BUY"   or "SELL"
# Offset from best ask/bid so the limit order sits passively and doesn't fill
TEST_LIMIT_OFFSET   = -0.10     # -0.10 means bid $0.10 below best ask

# How many recent ticks to show on the plot
HISTORY_TICKS = 50


# ── Step 1: Connect and check the case ────────────────────────────────────────

client = RITClient()

print("=" * 55)
print("STEP 1 — Connecting to RIT")
print("=" * 55)

try:
    case = client.get_case()
    print(f"  Case name : {case.get('name', 'N/A')}")
    print(f"  Status    : {case.get('status', 'N/A')}")
    print(f"  Period    : {case.get('period', 'N/A')}")
    print(f"  Tick      : {case.get('tick', 'N/A')}")
    print(f"  Raw response: {case}")
except Exception as e:
    print(f"  ERROR: Could not connect to RIT — {e}")
    print("  Make sure RIT is open and the API key in rit_client.py is correct.")
    sys.exit(1)

print()


# ── Step 2: Trader info ────────────────────────────────────────────────────────

print("=" * 55)
print("STEP 2 — Trader Account")
print("=" * 55)

trader = client.get_trader()
print(f"  Raw response: {trader}")
print(f"  Cash        : ${trader.get('cash', 0):,.2f}")
print(f"  Buying Power: ${trader.get('buying_power', 0):,.2f}")
print(f"  NLV         : ${trader.get('nlv', 0):,.2f}")
print()


# ── Step 3: Securities table ───────────────────────────────────────────────────

print("=" * 55)
print("STEP 3 — Available Securities")
print("=" * 55)

securities = client.get_securities()
if securities:
    print(f"  Raw first security: {securities[0]}")   # shows all field names RIT actually uses
    print()
    print(f"  {'TICKER':<10} {'TYPE':<10} {'LAST':>8} {'BID':>8} {'ASK':>8} {'POSITION':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for s in securities:
        print(
            f"  {s.get('ticker',''):<10} {s.get('type',''):<10}"
            f" {s.get('last', 0):>8.2f} {s.get('bid', 0):>8.2f}"
            f" {s.get('ask', 0):>8.2f} {s.get('position', 0):>10}"
        )
print()


# ── Step 4: Order book ─────────────────────────────────────────────────────────

print("=" * 55)
print(f"STEP 4 — Order Book for {TICKER}")
print("=" * 55)

book = client.get_order_book(TICKER)
bids = book.get("bids", [])
asks = book.get("asks", [])

print(f"  Raw book keys: {list(book.keys())}")
if bids:
    print(f"  Raw first bid entry: {bids[0]}")   # shows exact field names
if asks:
    print(f"  Raw first ask entry: {asks[0]}")
print()

print(f"  {'BID PRICE':>10} {'BID QTY':>10}    {'ASK PRICE':>10} {'ASK QTY':>10}")
print(f"  {'-'*10} {'-'*10}    {'-'*10} {'-'*10}")
for i in range(min(5, max(len(bids), len(asks)))):
    bid_price = f"{bids[i]['price']:.2f}"    if i < len(bids) else ""
    bid_qty   = f"{bids[i]['quantity']}"     if i < len(bids) else ""
    ask_price = f"{asks[i]['price']:.2f}"    if i < len(asks) else ""
    ask_qty   = f"{asks[i]['quantity']}"     if i < len(asks) else ""
    print(f"  {bid_price:>10} {bid_qty:>10}    {ask_price:>10} {ask_qty:>10}")
print()


# ── Step 5: Price history + plot ───────────────────────────────────────────────

print("=" * 55)
print(f"STEP 5 — Price History for {TICKER}")
print("=" * 55)

history = client.get_price_history(TICKER)

if not history:
    print(f"  No history yet for {TICKER} — case may just be starting, skipping plot.")
else:
    # Print a sample entry so we can confirm exact field names from RIT
    print(f"  Total bars returned : {len(history)}")
    print(f"  Raw first bar  : {history[0]}")
    print(f"  Raw last bar   : {history[-1]}")
    print()

    recent = history[-HISTORY_TICKS:]

    # Detect the tick/time field — RIT has used both "tick" and "period_tick"
    tick_key  = "tick"        if "tick"  in recent[0] else list(recent[0].keys())[0]
    close_key = "close"       if "close" in recent[0] else None
    high_key  = "high"        if "high"  in recent[0] else None
    low_key   = "low"         if "low"   in recent[0] else None

    if close_key is None:
        print(f"  WARNING: 'close' field not found in history. Fields are: {list(recent[0].keys())}")
        print("  Skipping plot — update the field names above once you see the raw output.")
    else:
        xs     = [bar[tick_key]  for bar in recent]
        closes = [bar[close_key] for bar in recent]
        highs  = [bar[high_key]  for bar in recent] if high_key else closes
        lows   = [bar[low_key]   for bar in recent] if low_key  else closes

        # 10-tick moving average
        ma10 = [None] * 9 + [
            sum(closes[i-9:i+1]) / 10 for i in range(9, len(closes))
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(f"{TICKER} — Last {len(recent)} Ticks", fontsize=14, fontweight="bold")

        ax1.fill_between(xs, lows, highs, alpha=0.15, color="steelblue", label="High/Low range")
        ax1.plot(xs, closes, color="steelblue", linewidth=1.5, label="Close")
        ax1.plot(xs, ma10,   color="orange",    linewidth=1.5, linestyle="--", label="10-tick MA")
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        spread = [h - l for h, l in zip(highs, lows)]
        ax2.bar(xs, spread, color="steelblue", alpha=0.6, label="High-Low range")
        ax2.set_ylabel("Range ($)")
        ax2.set_xlabel(tick_key)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        print("  Close the chart window to continue to the order step.")
        plt.show()

print()


# ── Step 6: Place a test order ────────────────────────────────────────────────

print("=" * 55)
print("STEP 6 — Placing a Test Order")
print("=" * 55)

try:
    if TEST_ORDER_TYPE == "MARKET":
        print(f"  Placing MARKET {TEST_ORDER_SIDE} {TEST_ORDER_QUANTITY} x {TEST_ORDER_TICKER}...")
        result = client.place_market_order(TEST_ORDER_TICKER, TEST_ORDER_SIDE, TEST_ORDER_QUANTITY)
        print(f"  Raw response: {result}")

    else:
        ask = client.best_ask(TEST_ORDER_TICKER)
        bid = client.best_bid(TEST_ORDER_TICKER)
        ref = ask if TEST_ORDER_SIDE == "BUY" else bid
        if ref is None:
            ref = client.get_security(TEST_ORDER_TICKER).get("last", 10.0)
        limit_price = round(ref + TEST_LIMIT_OFFSET, 2)

        print(f"  Placing LIMIT {TEST_ORDER_SIDE} {TEST_ORDER_QUANTITY} x {TEST_ORDER_TICKER}"
              f" @ ${limit_price:.2f}  (ref: ${ref:.2f}, offset: {TEST_LIMIT_OFFSET:+.2f})")
        result = client.place_limit_order(
            TEST_ORDER_TICKER, TEST_ORDER_SIDE, TEST_ORDER_QUANTITY, limit_price
        )
        print(f"  Raw response: {result}")

    order_id = result.get("order_id")
    print(f"  Order ID = {order_id}")

    # Wait so you can see the order sitting in RIT before we cancel it
    print()
    print("  >>> Check RIT now — you should see the order in the blotter. <<<")
    input("  Press ENTER when ready to cancel the test order and finish...")

    if order_id is not None:
        order = client.get_order(order_id)
        print(f"  Current status: {order.get('status')} "
              f"(filled {order.get('quantity_filled', 0)} / {order.get('quantity')})")

        if order.get("status") == "OPEN":
            cancel = client.cancel_order(order_id)
            print(f"  Cancelled: {cancel}")
        else:
            print(f"  Order already {order.get('status')} — nothing to cancel.")

except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 55)
print("All steps complete.")
print("=" * 55)

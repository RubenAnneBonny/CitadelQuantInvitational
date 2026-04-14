"""
RIT Quick Test Script
─────────────────────
Run this file to verify your connection to RIT, visualize live price data,
and place a small test order.

Requirements:
    pip install requests matplotlib

Make sure RIT is open and running before executing.
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Allow imports from the Competition folder one level up
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Competition"))
from rit_client import RITClient

# ── Settings ───────────────────────────────────────────────────────────────────
# Change this to any ticker that exists in your current RIT case
TICKER = "CRZY"

# Test order settings — keep quantity small so it doesn't affect your position much
TEST_ORDER_TICKER   = "CRZY"
TEST_ORDER_QUANTITY = 10
TEST_ORDER_TYPE     = "LIMIT"   # "MARKET" or "LIMIT"
TEST_ORDER_SIDE     = "BUY"     # "BUY"   or "SELL"
# Only used if TEST_ORDER_TYPE is "LIMIT" — set slightly below market to avoid filling
TEST_LIMIT_OFFSET   = -0.10     # e.g. -0.10 means bid $0.10 below best ask

# How many ticks of live data to collect before plotting
HISTORY_TICKS = 50


# ── Step 1: Connect and check the case ────────────────────────────────────────

client = RITClient()

print("=" * 50)
print("STEP 1 — Connecting to RIT")
print("=" * 50)

try:
    case = client.get_case()
    print(f"  Case name : {case.get('name', 'N/A')}")
    print(f"  Status    : {case.get('status', 'N/A')}")
    print(f"  Period    : {case.get('period', 'N/A')}")
    print(f"  Tick      : {case.get('tick', 'N/A')}")
except Exception as e:
    print(f"  ERROR: Could not connect to RIT — {e}")
    print("  Make sure RIT is open and the API key in rit_client.py is correct.")
    sys.exit(1)

print()


# ── Step 2: Print trader info ──────────────────────────────────────────────────

print("=" * 50)
print("STEP 2 — Trader Account Info")
print("=" * 50)

trader = client.get_trader()
print(f"  Trader     : {trader.get('first_name', '')} {trader.get('last_name', '')}")
print(f"  Cash       : ${trader.get('cash', 0):,.2f}")
print(f"  Buying Power: ${trader.get('buying_power', 0):,.2f}")
print(f"  NLV        : ${trader.get('nlv', 0):,.2f}")
print()


# ── Step 3: Print all available securities ────────────────────────────────────

print("=" * 50)
print("STEP 3 — Available Securities")
print("=" * 50)

securities = client.get_securities()
print(f"  {'TICKER':<10} {'TYPE':<10} {'LAST':>8} {'BID':>8} {'ASK':>8} {'POSITION':>10}")
print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for s in securities:
    print(
        f"  {s.get('ticker',''):<10} {s.get('type',''):<10}"
        f" {s.get('last', 0):>8.2f} {s.get('bid', 0):>8.2f}"
        f" {s.get('ask', 0):>8.2f} {s.get('position', 0):>10}"
    )
print()


# ── Step 4: Show order book for chosen ticker ─────────────────────────────────

print("=" * 50)
print(f"STEP 4 — Order Book for {TICKER}")
print("=" * 50)

book = client.get_order_book(TICKER)
bids = book.get("bids", [])
asks = book.get("asks", [])

print(f"  {'BID PRICE':>10} {'BID QTY':>10}    {'ASK PRICE':>10} {'ASK QTY':>10}")
print(f"  {'-'*10} {'-'*10}    {'-'*10} {'-'*10}")
for i in range(max(len(bids), len(asks))):
    bid_price = f"{bids[i]['price']:.2f}" if i < len(bids) else ""
    bid_qty   = f"{bids[i]['quantity']}"  if i < len(bids) else ""
    ask_price = f"{asks[i]['price']:.2f}" if i < len(asks) else ""
    ask_qty   = f"{asks[i]['quantity']}"  if i < len(asks) else ""
    print(f"  {bid_price:>10} {bid_qty:>10}    {ask_price:>10} {ask_qty:>10}")
print()


# ── Step 5: Collect price history and plot ────────────────────────────────────

print("=" * 50)
print(f"STEP 5 — Plotting Price History for {TICKER}")
print("=" * 50)

history = client.get_price_history(TICKER)

if not history:
    print(f"  No price history available yet for {TICKER} (case may just be starting).")
else:
    ticks  = [bar["tick"]  for bar in history[-HISTORY_TICKS:]]
    closes = [bar["close"] for bar in history[-HISTORY_TICKS:]]
    highs  = [bar["high"]  for bar in history[-HISTORY_TICKS:]]
    lows   = [bar["low"]   for bar in history[-HISTORY_TICKS:]]

    # Simple 10-tick moving average (only plot once we have enough data)
    ma10 = []
    for i in range(len(closes)):
        if i < 9:
            ma10.append(None)
        else:
            ma10.append(sum(closes[i-9:i+1]) / 10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{TICKER} — Last {len(ticks)} Ticks", fontsize=14, fontweight="bold")

    # Price chart with high/low range
    ax1.fill_between(ticks, lows, highs, alpha=0.15, color="steelblue", label="High/Low range")
    ax1.plot(ticks, closes, color="steelblue", linewidth=1.5, label="Close")
    ax1.plot(ticks, ma10,   color="orange",    linewidth=1.5, linestyle="--", label="10-tick MA")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Spread chart (ask - bid) — pulled live for the last tick only
    # For historical spread we approximate as (high - low) / 2
    spread = [(h - l) for h, l in zip(highs, lows)]
    ax2.bar(ticks, spread, color="steelblue", alpha=0.6, label="High-Low range (spread proxy)")
    ax2.set_ylabel("Range ($)")
    ax2.set_xlabel("Tick")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"  Plotted {len(ticks)} ticks. Close the chart window to continue to the order step.")
    plt.show()

print()


# ── Step 6: Place a test order ────────────────────────────────────────────────

print("=" * 50)
print("STEP 6 — Placing a Test Order")
print("=" * 50)

try:
    if TEST_ORDER_TYPE == "MARKET":
        print(f"  Placing MARKET {TEST_ORDER_SIDE} {TEST_ORDER_QUANTITY} x {TEST_ORDER_TICKER}...")
        result = client.place_market_order(TEST_ORDER_TICKER, TEST_ORDER_SIDE, TEST_ORDER_QUANTITY)

    else:  # LIMIT
        # Pick a limit price that is unlikely to fill so we can safely cancel it
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

    order_id = result.get("order_id")
    print(f"  Order accepted! ID = {order_id}")
    print(f"  Full response: {result}")

    # Wait a moment then check the order status
    time.sleep(1)
    order = client.get_order(order_id)
    print(f"  Status after 1s: {order.get('status')} "
          f"(filled {order.get('quantity_filled', 0)} / {order.get('quantity')})")

    # Cancel the order if it's still open (keeps your book clean after testing)
    if order.get("status") == "OPEN":
        cancel = client.cancel_order(order_id)
        print(f"  Cancelled order {order_id}: {cancel.get('status')}")

except Exception as e:
    print(f"  ERROR placing order: {e}")

print()
print("=" * 50)
print("All steps complete.")
print("=" * 50)

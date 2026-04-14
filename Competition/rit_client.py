import requests
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
# Swap API_KEY to your actual key once you receive it from the competition organizers.
# BASE_URL points to the RIT server. If RIT is running on a different machine on your
# LAN, replace "localhost" with that machine's IP (e.g. "http://192.168.1.10:9999/v1").
API_KEY = "YOUR_API_KEY_HERE"
BASE_URL = "http://localhost:9999/v1"


class RITClient:
    """
    Client for the Rotman Interactive Trader (RIT) REST API.

    Create one instance at the top of your strategy script and reuse it:

        client = RITClient()

        # Check what tick the case is on
        print(client.get_case())

        # Buy 200 shares of BULL at market
        client.buy_market("BULL", 200)

        # Post a limit sell for BEAR at $14.50
        client.sell_limit("BEAR", 100, price=14.50)

        # Cancel everything if things go wrong
        client.cancel_all_orders()

    All methods raise requests.HTTPError on a non-2xx response from RIT,
    so wrap calls in try/except if you want to handle failures gracefully.
    """

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    # ── Internal helpers ───────────────────────────────────────────────────────
    # These are used internally by every public method. You won't need to call
    # them directly, but they handle attaching the API key and parsing JSON.

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        resp = self.session.get(f"{self.base_url}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint: str, params: Optional[dict] = None) -> dict:
        resp = self.session.post(f"{self.base_url}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, endpoint: str, params: Optional[dict] = None) -> dict:
        resp = self.session.delete(f"{self.base_url}{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Case / session ─────────────────────────────────────────────────────────

    def get_case(self) -> dict:
        """
        Returns the current state of the trading case.

        Call this at the top of your loop to know whether the case is running,
        what period you're in, and what tick it is.

        Returns a dict like:
            {
                "name":    "Case 1",
                "period":  1,          # which period of the case (1-indexed)
                "tick":    42,         # current tick within the period
                "ticks_per_period": 300,
                "status":  "ACTIVE",   # "ACTIVE", "PAUSED", or "STOPPED"
                "is_open": True        # False once the market closes
            }

        Example:
            case = client.get_case()
            if case["status"] != "ACTIVE":
                print("Market is not running yet")
        """
        return self._get("/case")

    def get_trader(self) -> dict:
        """
        Returns info about your trader account for the current case.

        Useful for monitoring how much buying power you have left and your
        overall P&L without looking up individual positions.

        Returns a dict like:
            {
                "trader_id":      "trader1",
                "first_name":     "John",
                "last_name":      "Doe",
                "nlv":            10243.50,   # net liquidation value (total portfolio worth)
                "cash":           5000.00,    # uninvested cash
                "buying_power":   5000.00,    # how much you can still spend
                "realized_pnl":   243.50,     # locked-in profit/loss from closed trades
                "unrealized_pnl": 0.00        # open position profit/loss at current prices
            }

        Example:
            trader = client.get_trader()
            print(f"Buying power remaining: ${trader['buying_power']:.2f}")
        """
        return self._get("/trader")

    # ── Securities ─────────────────────────────────────────────────────────────

    def get_securities(self) -> list:
        """
        Returns a list of all tradeable securities in the current case.

        Each entry shows the security's current price, your position in it,
        trading limits, and other metadata. Useful at startup to discover
        what tickers are available and their position limits.

        Returns a list of dicts, one per security, like:
            [
                {
                    "ticker":         "BULL",
                    "type":           "STOCK",   # STOCK, ETF, FUTURE, etc.
                    "last":           10.25,      # last traded price
                    "bid":            10.24,
                    "ask":            10.26,
                    "position":       150,        # your current net position (+ long, - short)
                    "position_limit": 1000,       # max absolute position allowed
                    "max_trade_size": 100,        # max shares per single order
                    "interest_rate":  0.0
                },
                ...
            ]

        Example:
            for sec in client.get_securities():
                print(sec["ticker"], sec["last"])
        """
        return self._get("/securities")

    def get_security(self, ticker: str) -> dict:
        """
        Returns info for a single security by ticker.

        Same fields as get_securities() but for just one ticker. Use this
        inside your loop when you only care about one specific asset.

        Args:
            ticker: the security symbol, e.g. "BULL" or "BEAR"

        Returns a dict (same structure as one entry from get_securities()):
            {
                "ticker":         "BULL",
                "type":           "STOCK",
                "last":           10.25,
                "bid":            10.24,
                "ask":            10.26,
                "position":       150,
                "position_limit": 1000,
                "max_trade_size": 100
            }

        Example:
            sec = client.get_security("BULL")
            print(f"BULL last price: {sec['last']}, my position: {sec['position']}")
        """
        data = self._get("/securities", params={"ticker": ticker})
        # RIT returns a list even for a single ticker — unwrap it
        if isinstance(data, list):
            return data[0]
        return data

    def get_order_book(self, ticker: str) -> dict:
        """
        Returns the full live order book for a security (all open bids and asks).

        Use this when you want to see depth beyond just the best bid/ask —
        e.g. to estimate how much liquidity is available at each price level,
        or to decide where to post a passive limit order.

        Args:
            ticker: the security symbol, e.g. "BULL"

        Returns a dict with two lists:
            {
                "bids": [
                    {"price": 10.24, "quantity": 500, "quantity_filled": 0},
                    {"price": 10.23, "quantity": 300, "quantity_filled": 0},
                    ...
                ],
                "asks": [
                    {"price": 10.26, "quantity": 400, "quantity_filled": 0},
                    ...
                ]
            }
        bids are sorted highest-first, asks are sorted lowest-first.

        Example:
            book = client.get_order_book("BULL")
            print(f"Best bid: {book['bids'][0]['price']}, size: {book['bids'][0]['quantity']}")
        """
        return self._get("/securities/book", params={"ticker": ticker})

    def get_price_history(self, ticker: str, period: Optional[int] = None) -> list:
        """
        Returns historical OHLC (open/high/low/close) price data for a security.

        Useful for computing moving averages, volatility, or any signal that
        needs past prices. By default returns all ticks for the entire case;
        pass a period number to restrict to a single period.

        Args:
            ticker: the security symbol, e.g. "BULL"
            period: (optional) period number to filter by, e.g. 1 for period 1 only

        Returns a list of dicts, one per tick, like:
            [
                {
                    "tick":   1,
                    "open":   10.00,
                    "high":   10.30,
                    "low":    9.95,
                    "close":  10.25,
                    "period": 1
                },
                ...
            ]
        Most recent tick is at the end of the list.

        Example:
            history = client.get_price_history("BULL")
            closes = [bar["close"] for bar in history]
            moving_avg = sum(closes[-10:]) / 10   # 10-tick moving average
        """
        params = {"ticker": ticker}
        if period is not None:
            params["period"] = period
        return self._get("/securities/history", params=params)

    # ── Orders ─────────────────────────────────────────────────────────────────

    def get_orders(self, status: Optional[str] = None) -> list:
        """
        Returns a list of your orders, optionally filtered by status.

        Args:
            status: (optional) one of:
                "OPEN"        — orders still resting in the book (not yet filled)
                "TRANSACTED"  — fully or partially filled orders
                "CANCELLED"   — orders you cancelled or that were rejected
                None          — returns all orders regardless of status

        Returns a list of order dicts like:
            [
                {
                    "order_id":        1024,
                    "ticker":          "BULL",
                    "type":            "LIMIT",
                    "action":          "BUY",
                    "quantity":        100,
                    "quantity_filled": 40,       # how many shares have traded so far
                    "price":           10.20,    # 0 for market orders
                    "status":          "OPEN"
                },
                ...
            ]

        Example:
            # Check how many open orders you currently have
            open_orders = client.get_orders(status="OPEN")
            print(f"{len(open_orders)} orders still resting in the book")
        """
        params = {}
        if status:
            params["status"] = status
        return self._get("/orders", params=params)

    def get_order(self, order_id: int) -> dict:
        """
        Returns details for a single order by its ID.

        The order_id is returned when you place an order via place_market_order,
        place_limit_order, buy_market, buy_limit, etc.

        Args:
            order_id: the integer ID of the order, e.g. 1024

        Returns a single order dict (same structure as entries from get_orders()):
            {
                "order_id":        1024,
                "ticker":          "BULL",
                "type":            "LIMIT",
                "action":          "BUY",
                "quantity":        100,
                "quantity_filled": 40,
                "price":           10.20,
                "status":          "OPEN"
            }

        Example:
            result = client.buy_limit("BULL", 100, price=10.20)
            oid = result["order_id"]
            # ... later ...
            order = client.get_order(oid)
            print(f"Filled {order['quantity_filled']} of {order['quantity']} shares")
        """
        return self._get(f"/orders/{order_id}")

    def place_market_order(self, ticker: str, action: str, quantity: int) -> dict:
        """
        Place a market order. Executes immediately at whatever price is available.

        Use this when you need to get in or out fast and don't care about the
        exact price. Be careful — wide spreads or thin books can cause bad fills.

        Args:
            ticker:   security symbol, e.g. "BULL"
            action:   "BUY" to go long / add to position, "SELL" to go short / reduce
            quantity: number of shares or contracts to trade

        Returns a dict confirming the order was received:
            {
                "order_id": 1025,
                "ticker":   "BULL",
                "type":     "MARKET",
                "action":   "BUY",
                "quantity": 100,
                "status":   "OPEN"   # will quickly move to "TRANSACTED"
            }

        Raises ValueError if action is not "BUY" or "SELL".

        Example:
            result = client.place_market_order("BULL", "BUY", 100)
            print(f"Order placed, ID: {result['order_id']}")
        """
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be 'BUY' or 'SELL', got '{action}'")
        return self._post("/orders", params={
            "ticker":   ticker.upper(),
            "type":     "MARKET",
            "action":   action,
            "quantity": quantity,
        })

    def place_limit_order(
        self, ticker: str, action: str, quantity: int, price: float
    ) -> dict:
        """
        Place a limit order. Only fills at your specified price or better.

        Use this to control your entry/exit price, post passive orders to earn
        the spread, or queue up trades in advance. The order sits in the book
        until filled or cancelled.

        Args:
            ticker:   security symbol, e.g. "BEAR"
            action:   "BUY" to bid below market, "SELL" to offer above market
            quantity: number of shares or contracts
            price:    your limit price — BUY orders fill at or below this,
                      SELL orders fill at or above this

        Returns a dict confirming the order was posted:
            {
                "order_id": 1026,
                "ticker":   "BEAR",
                "type":     "LIMIT",
                "action":   "SELL",
                "quantity": 50,
                "price":    14.50,
                "status":   "OPEN"
            }

        Raises ValueError if action is not "BUY" or "SELL".

        Example:
            result = client.place_limit_order("BEAR", "SELL", 50, price=14.50)
            print(f"Limit order posted at $14.50, ID: {result['order_id']}")
        """
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be 'BUY' or 'SELL', got '{action}'")
        return self._post("/orders", params={
            "ticker":   ticker.upper(),
            "type":     "LIMIT",
            "action":   action,
            "quantity": quantity,
            "price":    price,
        })

    def cancel_order(self, order_id: int) -> dict:
        """
        Cancel a single open order by its ID.

        Only works on orders with status "OPEN". Trying to cancel a filled or
        already-cancelled order will raise an HTTPError.

        Args:
            order_id: the integer ID of the order to cancel (from get_order or
                      the return value of place_limit_order / buy_limit etc.)

        Returns a dict confirming the cancellation:
            {"order_id": 1026, "status": "CANCELLED"}

        Example:
            result = client.buy_limit("BULL", 100, price=9.50)
            oid = result["order_id"]
            # Change of plans — remove it
            client.cancel_order(oid)
        """
        return self._delete(f"/orders/{order_id}")

    def cancel_all_orders(self) -> list[dict]:
        """
        Cancel every open order you currently have, across all tickers.

        Useful as an emergency kill-switch or end-of-period cleanup. Silently
        skips orders that fail to cancel (e.g. already filled) and includes
        error info in the return list for those.

        Returns a list of results, one per open order that was attempted:
            [
                {"order_id": 1026, "status": "CANCELLED"},
                {"order_id": 1027, "status": "CANCELLED"},
                {"order_id": 1028, "error": "404 Client Error: ..."},  # already filled
                ...
            ]

        Example:
            # Clean up at end of period
            cancelled = client.cancel_all_orders()
            print(f"Cancelled {len(cancelled)} orders")
        """
        open_orders = self.get_orders(status="OPEN")
        results = []
        for order in open_orders:
            try:
                results.append(self.cancel_order(order["order_id"]))
            except requests.HTTPError as e:
                results.append({"order_id": order["order_id"], "error": str(e)})
        return results

    # ── Convenience wrappers ───────────────────────────────────────────────────
    # Shorthand for the most common order types. Prefer these over calling
    # place_market_order / place_limit_order directly — they're less verbose.

    def buy_market(self, ticker: str, quantity: int) -> dict:
        """
        Market buy — immediately purchase `quantity` shares at the best available ask.

        Args:
            ticker:   security symbol, e.g. "BULL"
            quantity: number of shares to buy

        Returns the order confirmation dict (see place_market_order for full schema).

        Example:
            client.buy_market("BULL", 200)
        """
        return self.place_market_order(ticker, "BUY", quantity)

    def sell_market(self, ticker: str, quantity: int) -> dict:
        """
        Market sell — immediately sell `quantity` shares at the best available bid.

        Args:
            ticker:   security symbol, e.g. "BULL"
            quantity: number of shares to sell

        Returns the order confirmation dict (see place_market_order for full schema).

        Example:
            client.sell_market("BULL", 200)
        """
        return self.place_market_order(ticker, "SELL", quantity)

    def buy_limit(self, ticker: str, quantity: int, price: float) -> dict:
        """
        Limit buy — post a bid for `quantity` shares that only fills at or below `price`.

        The order rests in the book until it fills or you cancel it. Use this to
        avoid paying the ask spread, or to queue a buy at a target price.

        Args:
            ticker:   security symbol, e.g. "BEAR"
            quantity: number of shares to buy
            price:    maximum price you're willing to pay

        Returns the order confirmation dict (see place_limit_order for full schema).

        Example:
            client.buy_limit("BEAR", 100, price=9.80)
        """
        return self.place_limit_order(ticker, "BUY", quantity, price)

    def sell_limit(self, ticker: str, quantity: int, price: float) -> dict:
        """
        Limit sell — post an offer for `quantity` shares that only fills at or above `price`.

        Use this to exit a position at a target price, or to earn the spread by
        posting on the ask side.

        Args:
            ticker:   security symbol, e.g. "BEAR"
            quantity: number of shares to sell
            price:    minimum price you're willing to accept

        Returns the order confirmation dict (see place_limit_order for full schema).

        Example:
            client.sell_limit("BEAR", 50, price=14.50)
        """
        return self.place_limit_order(ticker, "SELL", quantity, price)

    # ── Positions ──────────────────────────────────────────────────────────────

    def get_positions(self) -> list:
        """
        Returns your current position in every security.

        This is the same data as get_securities() — RIT includes position info
        directly on each security object. Useful for a quick snapshot of your
        entire book.

        Returns a list of security dicts, each including:
            {
                "ticker":         "BULL",
                "position":       150,    # positive = long, negative = short, 0 = flat
                "position_limit": 1000,   # max absolute position allowed by the case
                "last":           10.25,
                ...
            }

        Example:
            for sec in client.get_positions():
                if sec["position"] != 0:
                    print(f"{sec['ticker']}: {sec['position']} shares @ last {sec['last']}")
        """
        return self._get("/securities")

    def get_position(self, ticker: str) -> int:
        """
        Returns your net position in a single security as an integer.

        Positive = long (you own shares), negative = short (you owe shares), 0 = flat.
        This is the quickest way to check one ticker's position without parsing
        the full securities list.

        Args:
            ticker: the security symbol, e.g. "BULL"

        Returns an int, e.g. 150 (long 150 shares) or -50 (short 50 shares).

        Example:
            pos = client.get_position("BULL")
            if pos > 500:
                print("Getting close to position limit, slow down buying")
        """
        data = self.get_security(ticker)
        return data.get("position", 0)

    def get_pnl(self) -> dict:
        """
        Returns a summary of your profit and loss.

        - realized_pnl:   profit/loss already locked in from trades you've closed
        - unrealized_pnl: profit/loss on positions still open (changes every tick)
        - nlv:            net liquidation value — total portfolio worth if you closed everything

        Returns:
            {
                "realized_pnl":   243.50,
                "unrealized_pnl": -12.00,
                "nlv":            10231.50
            }

        Example:
            pnl = client.get_pnl()
            print(f"Total NLV: ${pnl['nlv']:.2f}  (realized: ${pnl['realized_pnl']:.2f})")
        """
        trader = self.get_trader()
        return {
            "realized_pnl":   trader.get("realized_pnl", 0),
            "unrealized_pnl": trader.get("unrealized_pnl", 0),
            "nlv":            trader.get("nlv", 0),
        }

    # ── Utility ────────────────────────────────────────────────────────────────

    def mid_price(self, ticker: str) -> Optional[float]:
        """
        Returns the mid price — the average of the best bid and best ask.

        The mid is a clean reference price that sits between what buyers are
        willing to pay and what sellers are asking. Useful as a fair-value
        estimate when computing signals or deciding limit order placement.

        Args:
            ticker: the security symbol, e.g. "BULL"

        Returns a float (e.g. 10.25) or None if the book is empty on either side.

        Example:
            mid = client.mid_price("BULL")
            if mid:
                client.buy_limit("BULL", 100, price=mid - 0.02)  # bid 2 cents below mid
        """
        book = self.get_order_book(ticker)
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if bids and asks:
            return (bids[0]["price"] + asks[0]["price"]) / 2
        return None

    def best_bid(self, ticker: str) -> Optional[float]:
        """
        Returns the highest price any buyer is currently willing to pay.

        Selling at or below this price will get you an immediate fill.
        Returns None if there are no bids in the book.

        Args:
            ticker: the security symbol, e.g. "BULL"

        Returns a float (e.g. 10.24) or None if no bids exist.

        Example:
            bid = client.best_bid("BULL")
            if bid and bid > my_target_sell_price:
                client.sell_market("BULL", 100)
        """
        book = self.get_order_book(ticker)
        bids = book.get("bids", [])
        return bids[0]["price"] if bids else None

    def best_ask(self, ticker: str) -> Optional[float]:
        """
        Returns the lowest price any seller is currently willing to accept.

        Buying at or above this price will get you an immediate fill.
        Returns None if there are no asks in the book.

        Args:
            ticker: the security symbol, e.g. "BULL"

        Returns a float (e.g. 10.26) or None if no asks exist.

        Example:
            ask = client.best_ask("BULL")
            if ask and ask < my_target_buy_price:
                client.buy_market("BULL", 100)
        """
        book = self.get_order_book(ticker)
        asks = book.get("asks", [])
        return asks[0]["price"] if asks else None

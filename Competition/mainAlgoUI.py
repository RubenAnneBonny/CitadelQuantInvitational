import threading
import tkinter as tk
from tkinter import font
from rit_client import RITClient

# ── PnL tracker ────────────────────────────────────────────────────────────────

class PnLTracker:
    """
    Records every buy/sell made by one function and calculates its PnL.

    Realized PnL is locked in whenever a position is reduced or closed.
    Unrealized PnL is the open position valued at the current market price.
    """

    def __init__(self):
        self.realized   = 0.0
        self._positions = {}   # ticker -> {"qty": int, "avg_cost": float}

    def record(self, ticker: str, action: str, quantity: int, price: float,
               transaction_cost: float = 0.0):
        """Call this every time the function places an order."""
        if ticker not in self._positions:
            self._positions[ticker] = {"qty": 0, "avg_cost": 0.0}

        pos        = self._positions[ticker]
        qty        = pos["qty"]
        signed_qty = quantity if action == "BUY" else -quantity

        if qty == 0:
            # Opening a brand new position
            pos["qty"]      = signed_qty
            pos["avg_cost"] = price

        elif (qty > 0 and signed_qty > 0) or (qty < 0 and signed_qty < 0):
            # Adding to an existing position in the same direction → update average cost
            total_cost      = abs(qty) * pos["avg_cost"] + abs(signed_qty) * price
            pos["qty"]     += signed_qty
            pos["avg_cost"] = total_cost / abs(pos["qty"])

        else:
            # Reducing, closing, or reversing the position
            if abs(signed_qty) <= abs(qty):
                # Partial or full close — lock in realized PnL
                if qty > 0:
                    self.realized += abs(signed_qty) * (price - pos["avg_cost"])
                else:
                    self.realized += abs(signed_qty) * (pos["avg_cost"] - price)
                pos["qty"] += signed_qty
                if pos["qty"] == 0:
                    pos["avg_cost"] = 0.0
            else:
                # Full close + reversal to the other side
                if qty > 0:
                    self.realized += abs(qty) * (price - pos["avg_cost"])
                else:
                    self.realized += abs(qty) * (pos["avg_cost"] - price)
                pos["qty"]      = signed_qty + qty
                pos["avg_cost"] = price

        # Transaction cost is always a drag — subtract it regardless of direction
        self.realized -= abs(quantity) * transaction_cost

    def unrealized(self, securities: dict) -> float:
        """
        Unrealized PnL using the securities dict from RIT.

        Matches RIT's own mark-to-market convention:
          - Long positions are marked to bid (what you'd receive if you sold now)
          - Short positions are marked to ask (what you'd pay to cover now)
        """
        total = 0.0
        for ticker, pos in self._positions.items():
            if pos["qty"] == 0 or ticker not in securities:
                continue
            sec     = securities[ticker]
            current = sec.get("bid" if pos["qty"] > 0 else "ask", sec.get("last", 0.0))
            total  += pos["qty"] * (current - pos["avg_cost"])
        return total

    def total(self, securities: dict) -> float:
        return self.realized + self.unrealized(securities)


# ── Tracked client wrapper ─────────────────────────────────────────────────────

class TrackedClient:
    """
    Wraps RITClient so every order placed through it is recorded in a PnLTracker.
    All other RITClient methods (get_case, get_securities, etc.) pass through unchanged.

    For market orders the fill price is estimated from the current bid/ask.
    For limit orders the limit price is used.
    """

    def __init__(self, base_client: RITClient, tracker: PnLTracker):
        self._client    = base_client
        self._tracker   = tracker
        self._securities = {}   # updated each tick before the function runs

    def update_securities(self, securities: dict):
        self._securities = securities

    # ── Intercept order methods ────────────────────────────────────────────────

    def _tc(self, ticker: str) -> float:
        """Returns the per-share trading fee for a ticker (0 if unknown)."""
        return self._securities.get(ticker.upper(), {}).get("trading_fee", 0.0)

    def _fill_price(self, result: dict, ticker: str, action: str) -> float:
        """
        Best-effort fill price. Uses the actual price from the API response if
        present, otherwise falls back to the current bid/ask from securities.
        """
        if result.get("price"):
            return float(result["price"])
        sec = self._securities.get(ticker.upper(), {})
        return sec.get("ask", 0.0) if action == "BUY" else sec.get("bid", 0.0)

    def buy_market(self, ticker: str, quantity: int) -> dict:
        result = self._client.buy_market(ticker, quantity)
        self._tracker.record(ticker.upper(), "BUY", quantity,
                             self._fill_price(result, ticker, "BUY"), self._tc(ticker))
        return result

    def sell_market(self, ticker: str, quantity: int) -> dict:
        result = self._client.sell_market(ticker, quantity)
        self._tracker.record(ticker.upper(), "SELL", quantity,
                             self._fill_price(result, ticker, "SELL"), self._tc(ticker))
        return result

    def buy_limit(self, ticker: str, quantity: int, price: float) -> dict:
        result = self._client.buy_limit(ticker, quantity, price)
        self._tracker.record(ticker.upper(), "BUY", quantity, price, self._tc(ticker))
        return result

    def sell_limit(self, ticker: str, quantity: int, price: float) -> dict:
        result = self._client.sell_limit(ticker, quantity, price)
        self._tracker.record(ticker.upper(), "SELL", quantity, price, self._tc(ticker))
        return result

    def place_market_order(self, ticker: str, action: str, quantity: int) -> dict:
        result = self._client.place_market_order(ticker, action, quantity)
        action = action.upper()
        self._tracker.record(ticker.upper(), action, quantity,
                             self._fill_price(result, ticker, action), self._tc(ticker))
        return result

    def place_limit_order(self, ticker: str, action: str, quantity: int, price: float) -> dict:
        result = self._client.place_limit_order(ticker, action, quantity, price)
        self._tracker.record(ticker.upper(), action.upper(), quantity, price, self._tc(ticker))
        return result

    # ── Proxy everything else to the base client ───────────────────────────────

    def __getattr__(self, name):
        return getattr(self._client, name)


# ── Strategy functions ─────────────────────────────────────────────────────────

def spread(securities, ritClient) -> bool:
    security = securities["CRZY"]
    diff = (security["ask"] - security["bid"]) * 100
    ritClient.buy_market("CRZY", diff)
    return False

def tame_spread(securities, ritClient) -> bool:
    security = securities["TAME"]
    diff = (security["ask"] - security["bid"]) * 100
    ritClient.sell_market("TAME", diff)
    return False

# ── Function wrapper ───────────────────────────────────────────────────────────

class Function:
    def __init__(self, func, stop_value: float = 0.0):
        self.func            = func
        self.name            = func.__name__
        self.on              = True   # auto-cycle state: True=run this tick, False=cooldown
        self.off_ticks       = 0      # how many ticks spent in cooldown
        self.no_ticks        = False  # kept for compatibility
        self.disabled        = False  # user kill-switch: True=never run until re-enabled
        self.pnl_history     = []     # rolling PnL history for the sparkline (max 150 pts)
        self.tracker         = PnLTracker()
        self.tracked_client  = None   # set after client is created below
        # ── Drawdown stop-loss ─────────────────────────────────────────────────
        self.stop_value      = stop_value  # max drawdown from peak allowed (0 = disabled)
        self.peak_pnl        = 0.0         # high-watermark PnL since last reset
        self.stop_order_ids  = {}          # ticker -> order_id of resting stop-limit order
        self.stopped_by_risk = False       # True = auto-stopped by drawdown rule

    def suggested_stop_value(self, k: float = 2.0) -> float:
        """Returns k * std-dev of historical drawdowns — call after a warm-up period."""
        import statistics
        if len(self.pnl_history) < 10:
            return 0.0
        peak = self.pnl_history[0]
        drawdowns = []
        for p in self.pnl_history:
            peak = max(peak, p)
            drawdowns.append(peak - p)
        return round(k * statistics.stdev(drawdowns), 2)

# ── Client + function registry ─────────────────────────────────────────────────

client = RITClient()

functions = [
    Function(spread,      stop_value=2.0),
    Function(tame_spread, stop_value=2.0),
]

# Wire up each function's tracked client now that `client` exists
for f in functions:
    f.tracked_client = TrackedClient(client, f.tracker)

# ── Algo loop ──────────────────────────────────────────────────────────────────

TICKS_OFF = 5

state = {
    "tick":       0,
    "running":    False,
    "status":     "Stopped",
    "securities": {},   # latest {ticker: security_dict}, used for unrealized PnL
}

def _stop_price(pos: dict, peak_pnl: float, stop_value: float, realized: float) -> float:
    """
    Price at which this position's PnL contribution hits the stop threshold.

    Works for both longs (qty > 0 → sell stop) and shorts (qty < 0 → buy stop).
    The formula solves for `price` such that:
        realized + qty * (price - avg_cost) == peak_pnl - stop_value
    """
    qty = pos["qty"]
    return pos["avg_cost"] + (peak_pnl - stop_value - realized) / qty


def _update_stop_orders(func: "Function", securities: dict):
    """
    Place (or cancel-and-replace) one resting limit stop order per open ticker.

    Called every tick when the function is healthy and stop_value > 0.
    Each order is sized to close the full position for that ticker.
    """
    if func.stop_value == 0 or not func.tracker._positions:
        return

    realized = func.tracker.realized

    for ticker, pos in func.tracker._positions.items():
        qty = pos["qty"]
        if qty == 0:
            continue

        # Cancel old stop order for this ticker (silently ignore if already gone)
        if ticker in func.stop_order_ids:
            try:
                func.tracked_client.cancel_order(func.stop_order_ids[ticker])
            except Exception:
                pass
            del func.stop_order_ids[ticker]

        # Compute unrealized of all OTHER tickers so stop_price is accurate
        other_unrealized = sum(
            pos2["qty"] * (
                securities.get(t2, {}).get("bid" if pos2["qty"] > 0 else "ask",
                                           securities.get(t2, {}).get("last", 0.0))
                - pos2["avg_cost"]
            )
            for t2, pos2 in func.tracker._positions.items()
            if t2 != ticker and pos2["qty"] != 0 and t2 in securities
        )

        stop_p = _stop_price(
            pos,
            func.peak_pnl,
            func.stop_value,
            realized + other_unrealized,
        )

        if stop_p <= 0:
            continue

        try:
            if qty > 0:
                result = func.tracked_client.sell_limit(ticker, abs(qty), stop_p)
            else:
                result = func.tracked_client.buy_limit(ticker, abs(qty), stop_p)
            func.stop_order_ids[ticker] = result.get("order_id")
        except Exception:
            pass


def _trigger_stop(func: "Function", securities: dict, current_pnl: float):
    """
    Fire aggressive limit orders to close all positions, then disable the function.

    Called when the drawdown from the peak exceeds stop_value.
    """
    # Cancel all resting stop orders
    for ticker, oid in list(func.stop_order_ids.items()):
        try:
            func.tracked_client.cancel_order(oid)
        except Exception:
            pass
    func.stop_order_ids.clear()

    # Close every open position with an aggressive limit at the current best price
    for ticker, pos in func.tracker._positions.items():
        qty = pos["qty"]
        if qty == 0 or ticker not in securities:
            continue
        try:
            if qty > 0:
                close_price = securities[ticker].get("bid", 0.0)
                func.tracked_client.sell_limit(ticker, abs(qty), close_price)
            else:
                close_price = securities[ticker].get("ask", 0.0)
                func.tracked_client.buy_limit(ticker, abs(qty), close_price)
        except Exception:
            pass

    func.stopped_by_risk = True
    func.disabled        = True
    func.peak_pnl        = current_pnl   # reset so re-enable starts fresh


def algo_loop():
    pre_tick = -1

    while state["running"]:
        try:
            tick = client.get_case()["tick"]
        except Exception as e:
            state["status"] = f"Error: {e}"
            continue

        if tick == pre_tick:
            continue

        pre_tick        = tick
        state["tick"]   = tick
        state["status"] = "Running"

        try:
            securities = {s["ticker"]: s for s in client.get_securities()}
            state["securities"] = securities
        except Exception as e:
            state["status"] = f"Error fetching securities: {e}"
            continue

        for func in functions:
            # User kill-switch — completely independent of the auto-cycle
            if func.disabled:
                continue

            # Auto-cycle cooldown
            if not func.on:
                func.off_ticks += 1
                if func.off_ticks >= TICKS_OFF:
                    func.on        = True
                    func.off_ticks = 0
                else:
                    continue

            func.tracked_client.update_securities(securities)

            try:
                func.on = func.func(securities, func.tracked_client)
            except Exception as e:
                state["status"] = f"Error in {func.name}: {e}"

            # ── Drawdown stop-loss check ───────────────────────────────────────
            if func.stop_value > 0 and not func.stopped_by_risk:
                t = func.tracker.realized + func.tracker.unrealized(securities)
                func.peak_pnl = max(func.peak_pnl, t)
                if t <= func.peak_pnl - func.stop_value:
                    _trigger_stop(func, securities, t)
                else:
                    _update_stop_orders(func, securities)

# ── Dashboard ──────────────────────────────────────────────────────────────────

class Dashboard(tk.Tk):
    BG       = "#1e1e2e"
    PANEL    = "#2a2a3d"
    GREEN    = "#3ddc84"
    RED      = "#ff5c5c"
    DARK_RED = "#8b1a1a"
    PINK     = "#e91e8c"   # risk-stop state
    AMBER    = "#ffc44d"
    TEXT     = "#e0e0e0"
    SUBTEXT  = "#888888"
    POS_PNL  = "#3ddc84"   # positive PnL
    NEG_PNL  = "#ff5c5c"   # negative PnL
    BTN_ON   = "#3ddc84"
    BTN_OFF  = "#ff5c5c"
    BTN_FG   = "#1e1e2e"

    def __init__(self):
        super().__init__()
        self.title("RIT Algo Dashboard")
        self.configure(bg=self.BG)
        self.resizable(False, False)

        title_font = font.Font(family="Segoe UI",  size=14, weight="bold")
        label_font = font.Font(family="Segoe UI",  size=10)
        mono_font  = font.Font(family="Consolas",  size=10)
        btn_font   = font.Font(family="Segoe UI",  size=9,  weight="bold")
        big_font   = font.Font(family="Segoe UI",  size=22, weight="bold")
        small_font = font.Font(family="Segoe UI",  size=8)
        pnl_font   = font.Font(family="Consolas",  size=9)
        total_font = font.Font(family="Segoe UI",  size=12, weight="bold")

        # ── Header ─────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=self.BG, pady=12)
        header.pack(fill="x", padx=20)

        tk.Label(header, text="RIT Algo Dashboard",
                 font=title_font, bg=self.BG, fg=self.TEXT).pack(side="left")

        self.run_btn = tk.Button(
            header, text="START", width=10,
            font=btn_font, relief="flat", cursor="hand2",
            command=self._toggle_algo
        )
        self.run_btn.pack(side="right", padx=4)
        self._style_run_btn(running=False)

        # ── Status bar ─────────────────────────────────────────────────────────
        status_frame = tk.Frame(self, bg=self.PANEL, pady=10)
        status_frame.pack(fill="x", padx=20, pady=(0, 12))

        tick_col = tk.Frame(status_frame, bg=self.PANEL, padx=20)
        tick_col.pack(side="left")
        tk.Label(tick_col, text="TICK", font=small_font,
                 bg=self.PANEL, fg=self.SUBTEXT).pack()
        self.tick_label = tk.Label(tick_col, text="—",
                                   font=big_font, bg=self.PANEL, fg=self.AMBER)
        self.tick_label.pack()

        tk.Frame(status_frame, bg=self.SUBTEXT, width=1).pack(
            side="left", fill="y", padx=16, pady=6)

        status_col = tk.Frame(status_frame, bg=self.PANEL, padx=4)
        status_col.pack(side="left")
        tk.Label(status_col, text="STATUS", font=small_font,
                 bg=self.PANEL, fg=self.SUBTEXT).pack(anchor="w")
        self.status_label = tk.Label(status_col, text="Stopped",
                                     font=label_font, bg=self.PANEL, fg=self.RED)
        self.status_label.pack(anchor="w")

        # ── Function cards ──────────────────────────────────────────────────────
        tk.Label(self, text="Functions", font=label_font,
                 bg=self.BG, fg=self.SUBTEXT).pack(anchor="w", padx=20, pady=(0, 4))

        cards_frame = tk.Frame(self, bg=self.BG)
        cards_frame.pack(fill="x", padx=20, pady=(0, 12))

        self.card_widgets = []

        for func in functions:
            card = tk.Frame(cards_frame, bg=self.PANEL, pady=10, padx=14)
            card.pack(fill="x", pady=4)

            # Left: dot + name + state sub-text
            left = tk.Frame(card, bg=self.PANEL)
            left.pack(side="left", fill="x", expand=True)

            top_row = tk.Frame(left, bg=self.PANEL)
            top_row.pack(anchor="w")

            indicator = tk.Label(top_row, text="●", font=font.Font(size=14),
                                  bg=self.PANEL, fg=self.GREEN)
            indicator.pack(side="left", padx=(0, 8))

            name_lbl = tk.Label(top_row, text=func.name,
                                 font=mono_font, bg=self.PANEL, fg=self.TEXT)
            name_lbl.pack(side="left")

            state_lbl = tk.Label(left, text="ON  |  off_ticks: 0",
                                  font=small_font, bg=self.PANEL, fg=self.SUBTEXT, anchor="w")
            state_lbl.pack(anchor="w", padx=(22, 0))

            # PnL row
            pnl_lbl = tk.Label(left,
                                text="R: $0.00   U: $0.00   Total: $0.00",
                                font=pnl_font, bg=self.PANEL, fg=self.SUBTEXT, anchor="w")
            pnl_lbl.pack(anchor="w", padx=(22, 0), pady=(2, 0))

            # Right: toggle button
            btn = tk.Button(
                card, text="Turn OFF", width=9,
                font=btn_font, relief="flat", cursor="hand2",
                command=lambda f=func: self._toggle_function(f)
            )
            btn.configure(bg=self.BTN_OFF, fg=self.BTN_FG, activebackground=self.BTN_OFF)
            btn.pack(side="right", padx=(8, 0))

            # Sparkline canvas — sits between the PnL text and the button
            canvas = tk.Canvas(card, width=120, height=50,
                                bg=self.PANEL, highlightthickness=0)
            canvas.pack(side="right", padx=(8, 8))

            self.card_widgets.append((indicator, state_lbl, pnl_lbl, canvas, btn, func))

        # ── Model PnL total ─────────────────────────────────────────────────────
        tk.Frame(self, bg=self.SUBTEXT, height=1).pack(fill="x", padx=20, pady=(0, 10))

        total_frame = tk.Frame(self, bg=self.PANEL, pady=12, padx=16)
        total_frame.pack(fill="x", padx=20, pady=(0, 16))

        tk.Label(total_frame, text="MODEL PnL", font=small_font,
                 bg=self.PANEL, fg=self.SUBTEXT).pack(anchor="w")

        pnl_row = tk.Frame(total_frame, bg=self.PANEL)
        pnl_row.pack(fill="x", pady=(4, 0))

        # Left: label + number
        left = tk.Frame(pnl_row, bg=self.PANEL)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="Total (sum of all functions)", font=small_font,
                 bg=self.PANEL, fg=self.SUBTEXT).pack(anchor="w")
        self.total_lbl = tk.Label(left, text="$0.00",
                                   font=total_font, bg=self.PANEL, fg=self.SUBTEXT)
        self.total_lbl.pack(anchor="w")

        # Right: sparkline for model total
        self.total_canvas = tk.Canvas(pnl_row, width=200, height=50,
                                       bg=self.PANEL, highlightthickness=0)
        self.total_canvas.pack(side="right", padx=(8, 0))
        self.total_pnl_history: list[float] = []

        self._refresh_ui()

    # ── Button actions ──────────────────────────────────────────────────────────

    def _toggle_algo(self):
        if state["running"]:
            state["running"] = False
            state["status"]  = "Stopped"
        else:
            state["running"] = True
            threading.Thread(target=algo_loop, daemon=True).start()

    def _toggle_function(self, func: Function):
        if func.disabled:
            # Re-enable: reset auto-cycle and clear any risk-stop state
            func.disabled        = False
            func.stopped_by_risk = False
            func.stop_order_ids  = {}
            func.on              = True
            func.off_ticks       = 0
        else:
            # Disable: kill it and cancel any resting stop orders
            func.disabled = True
            for oid in list(func.stop_order_ids.values()):
                try:
                    client.cancel_order(oid)
                except Exception:
                    pass
            func.stop_order_ids.clear()

    # ── UI refresh ──────────────────────────────────────────────────────────────

    def _pnl_color(self, value: float) -> str:
        if value > 0:
            return self.POS_PNL
        if value < 0:
            return self.NEG_PNL
        return self.SUBTEXT

    def _refresh_ui(self):
        running = state["running"]
        self._style_run_btn(running)

        self.tick_label.configure(text=str(state["tick"]) if running else "—")

        status_text  = state["status"]
        status_color = (self.GREEN if status_text == "Running" else
                        self.AMBER if status_text.startswith("Error") else self.RED)
        self.status_label.configure(text=status_text, fg=status_color)

        securities = state["securities"]

        total_t = 0.0

        for (indicator, state_lbl, pnl_lbl, canvas, btn, func) in self.card_widgets:
            # On/off indicator — check risk-stop BEFORE disabled
            if func.stopped_by_risk:
                indicator.configure(fg=self.PINK)
                state_lbl.configure(
                    text=f"RISK STOP  |  peak: ${func.peak_pnl:+.2f}  stop: ${func.stop_value:.2f}"
                )
                btn.configure(text="Turn ON", bg=self.PINK, activebackground=self.PINK, fg=self.TEXT)
            elif func.disabled:
                indicator.configure(fg=self.DARK_RED)
                state_lbl.configure(text="OFF  |  manually disabled")
                btn.configure(text="Turn ON", bg=self.DARK_RED, activebackground=self.DARK_RED, fg=self.TEXT)
            elif func.on:
                indicator.configure(fg=self.GREEN)
                if func.stop_value > 0:
                    t_now = func.tracker.realized + func.tracker.unrealized(securities)
                    dist  = func.peak_pnl - t_now - func.stop_value
                    state_lbl.configure(
                        text=f"ON  |  stop: ${func.stop_value:.2f}  dist: ${dist:+.2f}"
                    )
                else:
                    state_lbl.configure(text="ON")
                btn.configure(text="Turn OFF", bg=self.GREEN, activebackground=self.GREEN, fg=self.BTN_FG)
            else:
                indicator.configure(fg=self.RED)
                state_lbl.configure(text=f"COOLDOWN  |  {TICKS_OFF - func.off_ticks} ticks left")
                btn.configure(text="Turn OFF", bg=self.RED, activebackground=self.RED, fg=self.BTN_FG)

            # Per-function PnL
            r = func.tracker.realized
            u = func.tracker.unrealized(securities)
            t = r + u
            total_t += t   # model total = direct sum of each function's total

            pnl_lbl.configure(
                text=f"R: ${r:+.2f}   U: ${u:+.2f}   Total: ${t:+.2f}",
                fg=self._pnl_color(t)
            )

            # Record history and redraw sparkline
            func.pnl_history.append(t)
            func.pnl_history = func.pnl_history[-150:]
            self._draw_sparkline(canvas, func.pnl_history, self._pnl_color(t))

        # Model total — just the sum of what each function shows above
        self.total_lbl.configure(text=f"${total_t:+.2f}", fg=self._pnl_color(total_t))

        self.total_pnl_history.append(total_t)
        self.total_pnl_history = self.total_pnl_history[-150:]
        self._draw_sparkline(self.total_canvas, self.total_pnl_history,
                             self._pnl_color(total_t), W=200)

        self.after(200, self._refresh_ui)

    def _draw_sparkline(self, canvas, history: list, color: str, W: int = 120):
        canvas.delete("all")
        H, pad = 50, 4

        if len(history) < 2:
            # Not enough data yet — just draw a dashed zero line
            canvas.create_line(pad, H // 2, W - pad, H // 2,
                                fill=self.SUBTEXT, dash=(2, 2))
            return

        lo, hi = min(history), max(history)
        span    = (hi - lo) if hi != lo else 1.0

        def fy(v):
            # Flip so positive PnL is drawn upward
            return H - pad - ((v - lo) / span) * (H - 2 * pad)

        def fx(i):
            return pad + i * (W - 2 * pad) / (len(history) - 1)

        # Zero reference line
        if lo <= 0 <= hi:
            y0 = fy(0)
        elif lo > 0:
            y0 = H - pad   # all positive — zero is below the chart
        else:
            y0 = pad        # all negative — zero is above the chart
        canvas.create_line(pad, y0, W - pad, y0, fill=self.SUBTEXT, dash=(2, 2))

        pts = [(fx(i), fy(v)) for i, v in enumerate(history)]

        # Filled area between the line and zero
        poly = [pad, y0] + [c for pt in pts for c in pt] + [W - pad, y0]
        canvas.create_polygon(poly, fill=color, outline="", stipple="gray25")

        # The sparkline itself
        line = [c for pt in pts for c in pt]
        canvas.create_line(line, fill=color, width=1.5, smooth=True)

    def _style_run_btn(self, running: bool):
        if running:
            self.run_btn.configure(text="STOP",  bg=self.RED,   fg=self.BTN_FG,
                                   activebackground=self.RED)
        else:
            self.run_btn.configure(text="START", bg=self.GREEN, fg=self.BTN_FG,
                                   activebackground=self.GREEN)


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()

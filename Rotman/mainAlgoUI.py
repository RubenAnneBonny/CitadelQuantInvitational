import threading
import tkinter as tk
from tkinter import font
from RotmanInteractiveTraderApi import (
    RotmanInteractiveTraderApi,
    OrderType,
    OrderAction,
)
from settings import settings

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
            pos["qty"]      = signed_qty
            pos["avg_cost"] = price

        elif (qty > 0 and signed_qty > 0) or (qty < 0 and signed_qty < 0):
            total_cost      = abs(qty) * pos["avg_cost"] + abs(signed_qty) * price
            pos["qty"]     += signed_qty
            pos["avg_cost"] = total_cost / abs(pos["qty"])

        else:
            if abs(signed_qty) <= abs(qty):
                if qty > 0:
                    self.realized += abs(signed_qty) * (price - pos["avg_cost"])
                else:
                    self.realized += abs(signed_qty) * (pos["avg_cost"] - price)
                pos["qty"] += signed_qty
                if pos["qty"] == 0:
                    pos["avg_cost"] = 0.0
            else:
                if qty > 0:
                    self.realized += abs(qty) * (price - pos["avg_cost"])
                else:
                    self.realized += abs(qty) * (pos["avg_cost"] - price)
                pos["qty"]      = signed_qty + qty
                pos["avg_cost"] = price

        self.realized -= abs(quantity) * transaction_cost

    def unrealized(self, securities: dict) -> float:
        """
        Unrealized PnL using the securities dict from RotmanInteractiveTraderApi.get_portfolio().

        Long positions are marked to bid, short positions to ask.
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
    Wraps RotmanInteractiveTraderApi so every order placed through it is recorded
    in a PnLTracker. All other API methods pass through unchanged via __getattr__.
    """

    def __init__(self, base_client: RotmanInteractiveTraderApi, tracker: PnLTracker):
        self._client    = base_client
        self._tracker   = tracker
        self._securities = {}   # updated each tick before the function runs

    def update_securities(self, securities: dict):
        self._securities = securities

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _tc(self, ticker: str) -> float:
        return self._securities.get(ticker.upper(), {}).get("trading_fee", 0.0)

    def _fill_price(self, result: dict, ticker: str, action: str) -> float:
        """Best-effort fill price from the order response, then bid/ask fallback."""
        if result.get("vwap"):
            return float(result["vwap"])
        if result.get("price"):
            return float(result["price"])
        sec = self._securities.get(ticker.upper(), {})
        return sec.get("ask", 0.0) if action == "BUY" else sec.get("bid", 0.0)

    # ── Order methods (match the old RITClient interface) ──────────────────────

    def buy_market(self, ticker: str, quantity: int) -> dict:
        result = self._client.place_order(ticker, OrderType.MARKET, quantity, OrderAction.BUY)
        self._tracker.record(ticker.upper(), "BUY", quantity,
                             self._fill_price(result, ticker, "BUY"), self._tc(ticker))
        return result

    def sell_market(self, ticker: str, quantity: int) -> dict:
        result = self._client.place_order(ticker, OrderType.MARKET, quantity, OrderAction.SELL)
        self._tracker.record(ticker.upper(), "SELL", quantity,
                             self._fill_price(result, ticker, "SELL"), self._tc(ticker))
        return result

    def buy_limit(self, ticker: str, quantity: int, price: float) -> dict:
        result = self._client.place_order(ticker, OrderType.LIMIT, quantity, OrderAction.BUY, price)
        self._tracker.record(ticker.upper(), "BUY", quantity, price, self._tc(ticker))
        return result

    def sell_limit(self, ticker: str, quantity: int, price: float) -> dict:
        result = self._client.place_order(ticker, OrderType.LIMIT, quantity, OrderAction.SELL, price)
        self._tracker.record(ticker.upper(), "SELL", quantity, price, self._tc(ticker))
        return result

    def place_market_order(self, ticker: str, action: str, quantity: int) -> dict:
        action_upper = action.upper()
        result = self._client.place_order(
            ticker, OrderType.MARKET, quantity, OrderAction[action_upper]
        )
        self._tracker.record(ticker.upper(), action_upper, quantity,
                             self._fill_price(result, ticker, action_upper), self._tc(ticker))
        return result

    def place_limit_order(self, ticker: str, action: str, quantity: int, price: float) -> dict:
        action_upper = action.upper()
        result = self._client.place_order(
            ticker, OrderType.LIMIT, quantity, OrderAction[action_upper], price
        )
        self._tracker.record(ticker.upper(), action_upper, quantity, price, self._tc(ticker))
        return result

    # ── Proxy everything else to the base client ───────────────────────────────

    def __getattr__(self, name):
        return getattr(self._client, name)


# ── Strategy functions ─────────────────────────────────────────────────────────

def spread(securities, ritClient) -> bool:
    security = securities["CRZY"]
    diff = (security["ask"] - security["bid"]) * 200
    ritClient.buy_market("CRZY", diff)
    return False

def tame_spread(securities, ritClient) -> bool:
    security = securities["TAME"]
    diff = (security["ask"] - security["bid"]) * 200
    ritClient.sell_market("TAME", diff)
    return False

# ── Function wrapper ───────────────────────────────────────────────────────────

class Function:
    def __init__(self, func, stop_value: float = 0.0):
        self.func            = func
        self.name            = func.__name__
        self.on              = True
        self.off_ticks       = 0
        self.no_ticks        = False
        self.disabled        = False
        self.pnl_history     = []
        self.tracker         = PnLTracker()
        self.tracked_client  = None   # set after client is created below
        self.stop_value      = stop_value
        self.peak_pnl        = 0.0
        self.stopped_by_risk = False

    def suggested_stop_value(self, k: float = 2.0) -> float:
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

client = RotmanInteractiveTraderApi(
    api_key=settings["api_key"],
    api_host=settings["api_host"],
)

functions = [
    Function(spread,      stop_value=10.0),
    Function(tame_spread, stop_value=10.0),
]

for f in functions:
    f.tracked_client = TrackedClient(client, f.tracker)

# ── Algo loop ──────────────────────────────────────────────────────────────────

TICKS_OFF = 5

state = {
    "tick":       0,
    "running":    False,
    "status":     "Stopped",
    "securities": {},
}

def _trigger_stop(func: "Function", securities: dict, current_pnl: float):
    """Close all positions with limit orders then disable the function."""
    for ticker, pos in list(func.tracker._positions.items()):
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
    func.peak_pnl        = current_pnl


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
            # get_portfolio() returns dict[ticker, Security] directly
            securities = client.get_portfolio()
            state["securities"] = securities
        except Exception as e:
            state["status"] = f"Error fetching securities: {e}"
            continue

        for func in functions:
            if func.disabled:
                continue

            if func.on:
                func.tracked_client.update_securities(securities)
                try:
                    func.on = func.func(securities, func.tracked_client)
                except Exception as e:
                    state["status"] = f"Error in {func.name}: {e}"
            else:
                func.off_ticks += 1
                if func.off_ticks >= TICKS_OFF:
                    func.on        = True
                    func.off_ticks = 0

            if func.stop_value > 0 and not func.stopped_by_risk:
                t = func.tracker.realized + func.tracker.unrealized(securities)
                func.peak_pnl = max(func.peak_pnl, t)
                if t <= func.peak_pnl - func.stop_value:
                    _trigger_stop(func, securities, t)

# ── Dashboard ──────────────────────────────────────────────────────────────────

class Dashboard(tk.Tk):
    BG       = "#1e1e2e"
    PANEL    = "#2a2a3d"
    GREEN    = "#3ddc84"
    RED      = "#ff5c5c"
    DARK_RED = "#8b1a1a"
    PINK     = "#e91e8c"
    AMBER    = "#ffc44d"
    TEXT     = "#e0e0e0"
    SUBTEXT  = "#888888"
    POS_PNL  = "#3ddc84"
    NEG_PNL  = "#ff5c5c"
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

            pnl_lbl = tk.Label(left,
                                text="R: $0.00   U: $0.00   Total: $0.00",
                                font=pnl_font, bg=self.PANEL, fg=self.SUBTEXT, anchor="w")
            pnl_lbl.pack(anchor="w", padx=(22, 0), pady=(2, 0))

            btn = tk.Button(
                card, text="Turn OFF", width=9,
                font=btn_font, relief="flat", cursor="hand2",
                command=lambda f=func: self._toggle_function(f)
            )
            btn.configure(bg=self.BTN_OFF, fg=self.BTN_FG, activebackground=self.BTN_OFF)
            btn.pack(side="right", padx=(8, 0))

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

        left = tk.Frame(pnl_row, bg=self.PANEL)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="Total (sum of all functions)", font=small_font,
                 bg=self.PANEL, fg=self.SUBTEXT).pack(anchor="w")
        self.total_lbl = tk.Label(left, text="$0.00",
                                   font=total_font, bg=self.PANEL, fg=self.SUBTEXT)
        self.total_lbl.pack(anchor="w")

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
            func.disabled        = False
            func.stopped_by_risk = False
            func.on              = True
            func.off_ticks       = 0
        else:
            func.disabled = True

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
                    room = t_now - (func.peak_pnl - func.stop_value)
                    state_lbl.configure(
                        text=f"ON  |  stop: ${func.stop_value:.2f}  room: ${room:+.2f}"
                    )
                else:
                    state_lbl.configure(text="ON")
                btn.configure(text="Turn OFF", bg=self.GREEN, activebackground=self.GREEN, fg=self.BTN_FG)
            else:
                indicator.configure(fg=self.RED)
                state_lbl.configure(text=f"COOLDOWN  |  {TICKS_OFF - func.off_ticks} ticks left")
                btn.configure(text="Turn OFF", bg=self.RED, activebackground=self.RED, fg=self.BTN_FG)

            r = func.tracker.realized
            u = func.tracker.unrealized(securities)
            t = r + u
            total_t += t

            pnl_lbl.configure(
                text=f"R: ${r:+.2f}   U: ${u:+.2f}   Total: ${t:+.2f}",
                fg=self._pnl_color(t)
            )

            func.pnl_history.append(t)
            func.pnl_history = func.pnl_history[-150:]
            self._draw_sparkline(canvas, func.pnl_history, self._pnl_color(t))

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
            canvas.create_line(pad, H // 2, W - pad, H // 2,
                                fill=self.SUBTEXT, dash=(2, 2))
            return

        lo, hi = min(history), max(history)
        span    = (hi - lo) if hi != lo else 1.0

        def fy(v):
            return H - pad - ((v - lo) / span) * (H - 2 * pad)

        def fx(i):
            return pad + i * (W - 2 * pad) / (len(history) - 1)

        if lo <= 0 <= hi:
            y0 = fy(0)
        elif lo > 0:
            y0 = H - pad
        else:
            y0 = pad
        canvas.create_line(pad, y0, W - pad, y0, fill=self.SUBTEXT, dash=(2, 2))

        pts = [(fx(i), fy(v)) for i, v in enumerate(history)]

        poly = [pad, y0] + [c for pt in pts for c in pt] + [W - pad, y0]
        canvas.create_polygon(poly, fill=color, outline="", stipple="gray25")

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

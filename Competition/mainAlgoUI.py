import threading
import tkinter as tk
from tkinter import font
from rit_client import RITClient

# ── Strategy functions ─────────────────────────────────────────────────────────

def spread(securities, ritClient: RITClient) -> bool:
    security = securities["CRZY"]
    diff = (security["ask"] - security["bid"]) * 100
    ritClient.buy_market("CRZY", diff)
    return False

def tame_spread(securities, ritClient: RITClient) -> bool:
    security = securities["TAME"]
    diff = (security["ask"] - security["bid"]) * 100
    ritClient.sell_market("TAME", diff)
    return False

# ── Function wrapper ───────────────────────────────────────────────────────────

class Function:
    def __init__(self, func):
        self.func     = func
        self.name     = func.__name__
        self.on       = True
        self.off_ticks = 0
        self.no_ticks  = False   # True = stay off forever (manual disable)

# ── Register functions here ────────────────────────────────────────────────────

functions = [
    Function(spread),
    Function(tame_spread),
]

# ── Algo loop (runs in background thread) ─────────────────────────────────────

TICKS_OFF = 5
client    = RITClient()

# Shared state written by the algo thread, read by the UI thread
state = {
    "tick":    0,
    "running": False,
    "status":  "Stopped",
}

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

        pre_tick       = tick
        state["tick"]  = tick
        state["status"] = "Running"

        try:
            securities = {s["ticker"]: s for s in client.get_securities()}
        except Exception as e:
            state["status"] = f"Error fetching securities: {e}"
            continue

        for func in functions:
            if not func.on:
                if not func.no_ticks:
                    func.off_ticks += 1
                if func.off_ticks >= TICKS_OFF:
                    func.on        = True
                    func.off_ticks = 0
                else:
                    continue

            try:
                func.on = func.func(securities, client)
            except Exception as e:
                state["status"] = f"Error in {func.name}: {e}"

# ── Dashboard UI ───────────────────────────────────────────────────────────────

class Dashboard(tk.Tk):
    # Colours
    BG          = "#1e1e2e"
    PANEL       = "#2a2a3d"
    GREEN       = "#3ddc84"
    RED         = "#ff5c5c"
    AMBER       = "#ffc44d"
    TEXT        = "#e0e0e0"
    SUBTEXT     = "#888888"
    BTN_ON      = "#3ddc84"
    BTN_OFF     = "#ff5c5c"
    BTN_FG      = "#1e1e2e"

    def __init__(self):
        super().__init__()
        self.title("RIT Algo Dashboard")
        self.configure(bg=self.BG)
        self.resizable(False, False)

        title_font  = font.Font(family="Segoe UI", size=14, weight="bold")
        label_font  = font.Font(family="Segoe UI", size=10)
        mono_font   = font.Font(family="Consolas",  size=10)
        btn_font    = font.Font(family="Segoe UI", size=9,  weight="bold")
        big_font    = font.Font(family="Segoe UI", size=22, weight="bold")
        small_font  = font.Font(family="Segoe UI", size=8)

        # ── Header ─────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=self.BG, pady=12)
        header.pack(fill="x", padx=20)

        tk.Label(header, text="RIT Algo Dashboard",
                 font=title_font, bg=self.BG, fg=self.TEXT).pack(side="left")

        # Start / Stop button
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

        # Tick counter
        tick_col = tk.Frame(status_frame, bg=self.PANEL, padx=20)
        tick_col.pack(side="left")
        tk.Label(tick_col, text="TICK", font=small_font,
                 bg=self.PANEL, fg=self.SUBTEXT).pack()
        self.tick_label = tk.Label(tick_col, text="—",
                                   font=big_font, bg=self.PANEL, fg=self.AMBER)
        self.tick_label.pack()

        # Divider
        tk.Frame(status_frame, bg=self.SUBTEXT, width=1).pack(
            side="left", fill="y", padx=16, pady=6)

        # Status text
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
        cards_frame.pack(fill="x", padx=20, pady=(0, 16))

        self.card_widgets = []   # list of (indicator, name_label, status_label, toggle_btn)

        for func in functions:
            card = tk.Frame(cards_frame, bg=self.PANEL, pady=10, padx=14)
            card.pack(fill="x", pady=4)

            # Coloured dot
            indicator = tk.Label(card, text="●", font=font.Font(size=14),
                                  bg=self.PANEL, fg=self.GREEN)
            indicator.pack(side="left", padx=(0, 10))

            # Name + sub-status
            text_col = tk.Frame(card, bg=self.PANEL)
            text_col.pack(side="left", fill="x", expand=True)
            name_lbl = tk.Label(text_col, text=func.name,
                                 font=mono_font, bg=self.PANEL, fg=self.TEXT, anchor="w")
            name_lbl.pack(anchor="w")
            sub_lbl = tk.Label(text_col, text="ON  |  off_ticks: 0",
                                font=small_font, bg=self.PANEL, fg=self.SUBTEXT, anchor="w")
            sub_lbl.pack(anchor="w")

            # Toggle button
            btn = tk.Button(
                card, text="Turn OFF", width=9,
                font=btn_font, relief="flat", cursor="hand2",
                command=lambda f=func: self._toggle_function(f)
            )
            btn.configure(bg=self.BTN_OFF, fg=self.BTN_FG, activebackground=self.BTN_OFF)
            btn.pack(side="right")

            self.card_widgets.append((indicator, sub_lbl, btn, func))

        # ── Start polling UI updates ────────────────────────────────────────────
        self._refresh_ui()

    # ── Button actions ──────────────────────────────────────────────────────────

    def _toggle_algo(self):
        if state["running"]:
            state["running"] = False
            state["status"]  = "Stopped"
        else:
            state["running"] = True
            t = threading.Thread(target=algo_loop, daemon=True)
            t.start()

    def _toggle_function(self, func: Function):
        if func.on or not func.no_ticks:
            # Turn OFF manually → set no_ticks so it won't auto-restart
            func.on       = False
            func.no_ticks = True
            func.off_ticks = 0
        else:
            # Turn ON manually
            func.on       = True
            func.no_ticks = False
            func.off_ticks = 0

    # ── UI refresh (polls every 200 ms) ────────────────────────────────────────

    def _refresh_ui(self):
        running = state["running"]
        self._style_run_btn(running)

        # Tick + status
        self.tick_label.configure(
            text=str(state["tick"]) if running else "—"
        )
        status_text  = state["status"]
        status_color = self.GREEN if status_text == "Running" else (
                        self.AMBER if status_text.startswith("Error") else self.RED)
        self.status_label.configure(text=status_text, fg=status_color)

        # Function cards
        for (indicator, sub_lbl, btn, func) in self.card_widgets:
            if func.on:
                indicator.configure(fg=self.GREEN)
                sub_lbl.configure(text=f"ON  |  off_ticks: {func.off_ticks}")
                btn.configure(text="Turn OFF", bg=self.BTN_OFF)
            else:
                indicator.configure(fg=self.RED)
                mode = "manual (no auto-restart)" if func.no_ticks else f"auto-restart in {TICKS_OFF - func.off_ticks} ticks"
                sub_lbl.configure(text=f"OFF  |  {mode}")
                btn.configure(text="Turn ON", bg=self.BTN_ON)

        self.after(200, self._refresh_ui)

    def _style_run_btn(self, running: bool):
        if running:
            self.run_btn.configure(text="STOP", bg="#ff5c5c", fg=self.BTN_FG,
                                   activebackground="#ff5c5c")
        else:
            self.run_btn.configure(text="START", bg=self.GREEN, fg=self.BTN_FG,
                                   activebackground=self.GREEN)


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()

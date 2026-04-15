"""
Kalman Signal Simulation
=========================
Tests KalmanBBBSignal on data.csv by replaying prices one tick at a time,
exactly as it would receive them live from the Rotman API.

No future data is used at any point. The filter only ever sees prices
up to and including the current tick.

Configure the section at the top, then run:
    python kalman_sim.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import KalmanBBBSignal from the same folder ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from kalman_signal import KalmanBBBSignal

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════════════════════

DATA_FILE   = os.path.join(os.path.dirname(__file__),
                           "../Competition/Alpha Testing V2/data.csv")
STOCK       = "BBB"        # column to run the filter on
WARMUP_TICKS = 0           # ticks fed to KalmanBBBSignal before streaming starts
                           # 0 = true cold start (filter sees nothing upfront)
                           # e.g. 50 = give the filter 50 ticks of history first

# Kalman noise parameters (passed straight to KalmanBBBSignal)
POS_NOISE   = 0.01
VEL_NOISE   = 5.0
OBS_NOISE   = 1.0
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(df: pd.DataFrame, stock: str) -> pd.DataFrame:
    """
    Replay prices from df[stock] one tick at a time.

    Returns a DataFrame with columns:
        tick        — row index
        price       — actual observed price
        position    — Kalman filtered price estimate
        velocity    — Kalman velocity estimate (price / tick)
        signal      — normalised velocity signal in [-1, 1]
    """
    prices = df[stock].values

    warmup = prices[:WARMUP_TICKS].tolist() if WARMUP_TICKS > 0 else None
    kf     = KalmanBBBSignal(
        warmup_prices=warmup,
        pos_noise=POS_NOISE,
        vel_noise=VEL_NOISE,
        obs_noise=OBS_NOISE,
            )

    records = []

    # If we used warmup, record those ticks first (filter already processed them)
    if WARMUP_TICKS > 0:
        # Re-run warmup through a temporary filter just to get per-tick estimates
        # for plotting — the real filter state was initialised from these
        tmp_kf = KalmanBBBSignal(
            pos_noise=POS_NOISE,
            vel_noise=VEL_NOISE,
            obs_noise=OBS_NOISE,
                    )
        for i in range(WARMUP_TICKS):
            sig = tmp_kf.update(prices[i])
            records.append({
                "tick":     i,
                "price":    prices[i],
                "position": tmp_kf.position,
                "velocity": tmp_kf.velocity,
                "signal":   sig,
                "phase":    "warmup",
            })

    # Stream the remaining ticks one at a time — no future data
    for i in range(WARMUP_TICKS, len(prices)):
        sig = kf.update(prices[i])
        records.append({
            "tick":     i,
            "price":    prices[i],
            "position": kf.position,
            "velocity": kf.velocity,
            "signal":   sig,
            "phase":    "live",
        })

    return pd.DataFrame(records)


def plot_results(results: pd.DataFrame, stock: str):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Kalman Filter Simulation — {stock}   "
        f"(warmup={WARMUP_TICKS} ticks, "
        f"vel_noise={VEL_NOISE}, pos_noise={POS_NOISE})",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    ticks    = results["tick"].values
    price    = results["price"].values
    position = results["position"].values
    velocity = results["velocity"].values
    signal   = results["signal"].values

    # Shade the warmup region on every subplot
    warmup_end = WARMUP_TICKS - 1 if WARMUP_TICKS > 0 else None

    # ── Panel 1: Price vs filtered position ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(ticks, price,    color="#888888", linewidth=0.8, label="Actual price", alpha=0.7)
    ax1.plot(ticks, position, color="#3ddc84", linewidth=1.4, label="Kalman position estimate")
    if warmup_end:
        ax1.axvspan(0, warmup_end, color="#ffc44d", alpha=0.08, label="Warmup")
    ax1.set_ylabel("Price")
    ax1.set_title("Actual Price vs Kalman Position Estimate")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Velocity estimate ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax2.fill_between(ticks, velocity, 0,
                     where=(velocity >= 0), color="#3ddc84", alpha=0.4, label="Upward")
    ax2.fill_between(ticks, velocity, 0,
                     where=(velocity < 0),  color="#ff5c5c", alpha=0.4, label="Downward")
    ax2.plot(ticks, velocity, color="#e0e0e0", linewidth=0.8, alpha=0.8)
    if warmup_end:
        ax2.axvspan(0, warmup_end, color="#ffc44d", alpha=0.08)
    ax2.set_ylabel("Velocity (price / tick)")
    ax2.set_title("Kalman Velocity Estimate")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: Normalised signal ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.axhline(0,   color="#555555", linewidth=0.8, linestyle="--")
    ax3.axhline( 1,  color="#555555", linewidth=0.5, linestyle=":", alpha=0.5)
    ax3.axhline(-1,  color="#555555", linewidth=0.5, linestyle=":", alpha=0.5)
    ax3.fill_between(ticks, signal, 0,
                     where=(signal >= 0), color="#3ddc84", alpha=0.4, label="Long")
    ax3.fill_between(ticks, signal, 0,
                     where=(signal < 0),  color="#ff5c5c", alpha=0.4, label="Short")
    ax3.plot(ticks, signal, color="#e0e0e0", linewidth=0.8, alpha=0.8)
    if warmup_end:
        ax3.axvspan(0, warmup_end, color="#ffc44d", alpha=0.08, label="Warmup")
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_ylabel("Signal")
    ax3.set_xlabel("Tick")
    ax3.set_title("Normalised Velocity Signal  [−1 = full short,  +1 = full long]")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    plt.show()


def print_summary(results: pd.DataFrame, stock: str):
    live = results[results["phase"] == "live"]
    print(f"\n{'='*60}")
    print(f"  Kalman Simulation — {stock}")
    print(f"{'='*60}")
    print(f"  Total ticks streamed : {len(live)}")
    print(f"  Warmup ticks         : {WARMUP_TICKS}")
    print(f"  Velocity noise       : {VEL_NOISE}")
    print(f"  Position noise       : {POS_NOISE}")
    print(f"\n  Velocity stats (live phase):")
    print(f"    Mean     : {live['velocity'].mean():+.4f}")
    print(f"    Std dev  : {live['velocity'].std():.4f}")
    print(f"    Max      : {live['velocity'].max():+.4f}")
    print(f"    Min      : {live['velocity'].min():+.4f}")
    print(f"\n  Signal stats (live phase):")
    print(f"    % time long  : {(live['signal'] > 0).mean()*100:.1f}%")
    print(f"    % time short : {(live['signal'] < 0).mean()*100:.1f}%")
    print(f"    % time flat  : {(live['signal'] == 0).mean()*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    if STOCK not in df.columns:
        raise ValueError(f"Column '{STOCK}' not found. Available: {list(df.columns)}")

    print(f"\nStreaming '{STOCK}' through Kalman filter ({len(df)} ticks)...")
    results = run_simulation(df, STOCK)

    print_summary(results, STOCK)
    plot_results(results, STOCK)

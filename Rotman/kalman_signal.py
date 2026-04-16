"""
Online Kalman Filter Signal
============================
Rewrite of ETF_KalmanBBB from alpha_test.ipynb for live/continuous data.

State vector: [position, velocity]
  - position: filtered price estimate
  - velocity: rate of change of price (the trading signal)

The transition model is constant-velocity:
    pos_{t+1} = pos_t + vel_t  + noise
    vel_{t+1} = vel_t          + noise   (high noise = velocity can change fast)

Only position is observed; velocity is inferred by the filter.

Setting velocity process noise high (vel_noise) makes the filter track
trend changes aggressively rather than over-smoothing them.

Usage:
    from RotmanInteractiveTraderApi import RotmanInteractiveTraderApi
    from settings import settings

    client = RotmanInteractiveTraderApi(
        api_key=settings["api_key"], api_host=settings["api_host"]
    )

    # At startup, seed with whatever history the API has
    history = client.get_history("BBB")
    warmup = [h["close"] for h in history]
    signal_gen = KalmanBBBSignal(warmup_prices=warmup)

    # Each tick, feed the latest price and get a signal in [-1, 1]
    latest_price = client.get_portfolio()["BBB"]["last"]
    signal = signal_gen.update(latest_price)

    # Raw estimates are also available after each update:
    print(signal_gen.position)   # filtered price
    print(signal_gen.velocity)   # raw velocity (units: price / tick)
"""

import numpy as np
from pykalman import KalmanFilter


class KalmanBBBSignal:
    """
    Incremental Kalman filter signal generator with position + velocity state.

    Parameters
    ----------
    warmup_prices : list or array, optional
        Historical prices used to initialise the filter state.
        More warmup → better velocity estimate at the start.
    pos_noise : float
        Process noise on position (how much the true price can jump per tick
        beyond what velocity predicts). Keep low — prices move smoothly.
    vel_noise : float
        Process noise on velocity (how fast the trend can change). Set high
        so the filter adapts quickly to momentum shifts.
    obs_noise : float
        Observation noise (measurement uncertainty on the price feed).
    """

    def __init__(
        self,
        warmup_prices=None,
        pos_noise: float = 0.01,
        vel_noise: float = 5.0,
        obs_noise: float = 1.0,
    ):
        # ── Build the Kalman filter (fixed parameters — no EM) ────────────────
        # State: [position, velocity]
        # Transition: [[1, 1],   — pos_{t+1} = pos_t + vel_t
        #              [0, 1]]   — vel_{t+1} = vel_t
        # Observation: [[1, 0]]  — we only see position

        self._kf = KalmanFilter(
            transition_matrices=np.array([[1.0, 1.0],
                                          [0.0, 1.0]]),
            observation_matrices=np.array([[1.0, 0.0]]),
            transition_covariance=np.array([[pos_noise, 0.0],
                                            [0.0,       vel_noise]]),
            observation_covariance=np.array([[obs_noise]]),
            initial_state_covariance=np.array([[pos_noise, 0.0],
                                               [0.0,       vel_noise]]),
            n_dim_state=2,
            n_dim_obs=1,
        )

        self._max_abs_vel = 0.0

        # Public attributes — updated every call to update()
        self.position: float = 0.0
        self.velocity: float = 0.0

        # ── Initialise state ──────────────────────────────────────────────────
        warmup = np.array(warmup_prices, dtype=float) if warmup_prices is not None else np.array([])

        if len(warmup) > 0:
            # Set initial state mean to [first_price, 0]
            self._kf.initial_state_mean = np.array([warmup[0], 0.0])

            # Run filter forward over warmup to get a good state estimate
            observations = warmup.reshape(-1, 1)
            means, covs = self._kf.filter(observations)

            self._state_mean = means[-1]   # shape (2,)
            self._state_cov  = covs[-1]    # shape (2, 2)

            self.position = float(means[-1][0])
            self.velocity = float(means[-1][1])

            # Seed normalisation max with warmup velocity estimates
            self._max_abs_vel = max(abs(m[1]) for m in means)
        else:
            self._kf.initial_state_mean = np.array([0.0, 0.0])
            self._state_mean = self._kf.initial_state_mean.copy()
            self._state_cov  = self._kf.initial_state_covariance.copy()

    def update(self, new_price: float) -> float:
        """
        Ingest one new price tick, update the position + velocity estimates,
        and return a normalised velocity signal in [-1, 1].

        A positive signal means the filter sees upward momentum (go long).
        A negative signal means downward momentum (go short).
        Magnitude reflects strength relative to recent velocity history.

        After calling this, `self.position` and `self.velocity` are updated
        with the latest raw Kalman estimates.

        Parameters
        ----------
        new_price : float
            Latest observed price.

        Returns
        -------
        float
            Normalised velocity signal in [-1, 1].
        """
        observation = np.array([new_price])

        new_mean, new_cov = self._kf.filter_update(
            self._state_mean,
            self._state_cov,
            observation,
        )

        self._state_mean = new_mean
        self._state_cov  = new_cov

        self.position = float(new_mean[0])
        self.velocity = float(new_mean[1])

        # Normalise velocity by the all-time max absolute velocity
        self._max_abs_vel = max(self._max_abs_vel, abs(self.velocity))
        if self._max_abs_vel == 0.0:
            return 0.0

        return float(np.clip(self.velocity / self._max_abs_vel, -1.0, 1.0))

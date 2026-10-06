"""
config.py - Central, validated configuration for the whole simulation.

Every "magic number" of the project lives here so that experiments are easy to
re-configure (PHASE 1).  Money is expressed in Indian rupees (INR), energy in
kWh and power in kW.  Because the time step is exactly 1 hour, kW == kWh/h and
we use the two interchangeably (documented assumption).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

HOURS_PER_DAY = 24
SCENARIOS = ("baseline", "selfish", "cooperative")


@dataclass
class Config:
    # ------------------------------------------------------------------ scale
    n_households: int = 15          # 10-20 households
    n_days: int = 7                 # simulation length (days)
    horizon: int = 24               # planning horizon of each agent (hours)
    seed: int = 42                  # master random seed (reproducibility)

    # ------------------------------------------------- time-of-use grid tariff
    offpeak_hours: tuple = (23, 0, 1, 2, 3, 4, 5)
    peak_hours: tuple = (18, 19, 20, 21, 22)
    price_offpeak: float = 4.0      # INR/kWh
    price_shoulder: float = 6.5     # INR/kWh
    price_peak: float = 10.0        # INR/kWh
    sell_price_ratio: float = 0.45  # grid buys our energy at 45 % of its sale price

    # --------------------------------------------------- household heterogeneity
    battery_capacity_range: tuple = (4.0, 14.0)    # kWh
    battery_c_rate: float = 0.3                    # max (dis)charge power = c_rate * capacity
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc_fraction: float = 0.10                 # depth-of-discharge limit
    initial_soc_fraction: float = 0.50
    solar_kwp_range: tuple = (1.0, 4.0)            # installed PV size
    demand_scale_range: tuple = (0.5, 1.5)         # multiplies the base demand curve
    solar_noise_std: float = 0.08                  # hourly multiplicative PV noise
    cloud_min: float = 0.55                        # worst daily cloud factor (shared weather)
    demand_noise_std: float = 0.15                 # hourly log-normal demand noise

    # ---------------------------------------------------- Simulated Annealing
    sa_alpha: float = 0.85                 # geometric cooling factor
    sa_iters_per_temp: int = 25            # moves tried at each temperature
    sa_max_iter: int = 600                 # hard iteration cap
    sa_patience: int = 300                 # stop after this many iterations w/o improvement
    sa_t_min_ratio: float = 0.01           # stop when T < ratio * T0
    sa_initial_accept_prob: float = 0.8    # used to calibrate T0
    sa_calibration_samples: int = 20

    # ------------------------------------------------------- objective weights
    degradation_cost: float = 0.05         # INR per kWh of battery throughput
    terminal_value_factor: float = 0.5     # value of leftover stored kWh = factor*eff*mean buy price
    violation_penalty: float = 1000.0      # INR per kWh of constraint violation
    price_blend_kappa: float = 0.5         # how much agents trust the public P2P clearing price
    coop_congestion_weight: float = 0.05   # cooperative: weight on community congestion price
    coop_peak_weight: float = 0.04         # cooperative: quadratic peak penalty (INR/kWh^2)

    # ------------------------------------------------------------- auction
    selfish_shade_range: tuple = (0.15, 0.35)   # strategic bid shading of selfish agents
    auction_k: float = 0.5                      # clearing price = ask + k*(bid-ask) of marginal pair

    def __post_init__(self) -> None:
        self.validate()

    # --------------------------------------------------------------- helpers
    def validate(self) -> None:
        if not 1 <= self.n_households <= 200:
            raise ValueError("n_households must be in [1, 200] (project default 10-20)")
        if self.n_days < 1:
            raise ValueError("n_days must be >= 1")
        if not 1 <= self.horizon <= 48:
            raise ValueError("horizon must be in [1, 48]")
        if not self.price_offpeak > 0 or not self.price_shoulder > 0 or not self.price_peak > 0:
            raise ValueError("prices must be positive")
        if not 0 < self.sell_price_ratio < 1:
            raise ValueError("sell_price_ratio must be in (0,1) so that grid buy price > sell price")
        for name in ("charge_efficiency", "discharge_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.min_soc_fraction < 1:
            raise ValueError("min_soc_fraction must be in [0, 1)")
        if not self.min_soc_fraction <= self.initial_soc_fraction <= 1:
            raise ValueError("initial_soc_fraction must be in [min_soc_fraction, 1]")
        if not 0 < self.sa_alpha < 1:
            raise ValueError("sa_alpha must be in (0, 1)")
        if self.sa_max_iter < 1 or self.sa_iters_per_temp < 1:
            raise ValueError("SA iteration counts must be >= 1")
        if not 0 <= self.auction_k <= 1:
            raise ValueError("auction_k must be in [0, 1]")

    def grid_buy_prices(self) -> np.ndarray:
        """Price at which households BUY from the grid, per hour of day (24,)."""
        prices = np.full(HOURS_PER_DAY, self.price_shoulder, dtype=float)
        prices[list(self.offpeak_hours)] = self.price_offpeak
        prices[list(self.peak_hours)] = self.price_peak
        return prices

    def grid_sell_prices(self) -> np.ndarray:
        """Price at which households SELL to the grid (always < buy price)."""
        return self.grid_buy_prices() * self.sell_price_ratio

    def to_dict(self) -> dict:
        return asdict(self)

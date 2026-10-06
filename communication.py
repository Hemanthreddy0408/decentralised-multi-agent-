"""
communication.py - The ONLY information that crosses agent boundaries.

Agents never hold references to each other.  What they may know about the
neighbourhood is exactly what is placed in ``PublicInfo`` (tariffs, past public
market prices, aggregate transformer load, and the surplus/deficit messages that
agents voluntarily broadcast).  Battery SOC, private valuations, true future
demand and the SA state are NEVER part of these objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from config import HOURS_PER_DAY


@dataclass(frozen=True)
class BroadcastMessage:
    """Public announcement: 'in this hour I have a surplus/deficit of X kWh'."""
    agent_id: int
    hour: int
    kind: str          # "surplus" or "deficit"
    quantity: float    # kWh


@dataclass
class PublicInfo:
    """Publicly observable market / grid information (the 'public bid information')."""
    hour: int
    grid_buy: np.ndarray                  # INR/kWh per hour of day (tariff is public)
    grid_sell: np.ndarray
    clearing_ema: np.ndarray              # smoothed past P2P clearing price per hour of day
    agg_load_ema: np.ndarray              # smoothed aggregate grid import per hour of day
    load_seen: np.ndarray                 # which hours of day have load history
    broadcasts: list = field(default_factory=list)

    @classmethod
    def initial(cls, grid_buy: np.ndarray, grid_sell: np.ndarray) -> "PublicInfo":
        return cls(
            hour=0,
            grid_buy=grid_buy.copy(),
            grid_sell=grid_sell.copy(),
            clearing_ema=0.5 * (grid_buy + grid_sell),   # prior: midpoint of the spread
            agg_load_ema=np.zeros(HOURS_PER_DAY),
            load_seen=np.zeros(HOURS_PER_DAY, dtype=bool),
        )

    def total_surplus(self) -> float:
        return sum(m.quantity for m in self.broadcasts if m.kind == "surplus")

    def total_deficit(self) -> float:
        return sum(m.quantity for m in self.broadcasts if m.kind == "deficit")

    def congestion_factor(self, hour: int, horizon: int) -> np.ndarray:
        """
        f_t = (aggregate neighbourhood load at hour t) / (mean load).  f_t > 1 means the
        neighbourhood is expected to be congested at that hour.  Hours without history
        get the neutral value 1.0.
        """
        factors = np.ones(horizon)
        if not self.load_seen.any():
            return factors
        mean_load = self.agg_load_ema[self.load_seen].mean()
        if mean_load <= 1e-9:
            return factors
        for k in range(horizon):
            h = (hour + k) % HOURS_PER_DAY
            if self.load_seen[h]:
                factors[k] = float(np.clip(self.agg_load_ema[h] / mean_load, 0.3, 3.0))
        return factors

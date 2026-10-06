"""results.py - Container for everything recorded during one simulation run."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    scenario: str
    config: dict
    agent_ids: list
    agent_tags: list
    solar: np.ndarray          # (T, N) kWh
    demand: np.ndarray
    soc: np.ndarray            # SOC at the end of each hour (kWh)
    charge_in: np.ndarray
    discharge_out: np.ndarray
    grid_import: np.ndarray
    grid_export: np.ndarray
    p2p_bought: np.ndarray
    p2p_sold: np.ndarray
    agent_cost: np.ndarray     # (T, N) INR per hour
    grid_buy_price: np.ndarray  # (T,)
    grid_sell_price: np.ndarray
    p2p_price: np.ndarray       # (T,) NaN when no trade
    initial_soc: np.ndarray     # (N,)
    final_soc: np.ndarray
    inventory_value_per_kwh: np.ndarray   # (N,) INR value of 1 kWh left in the battery
    sa_stats: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return self.solar.shape[0]

    @property
    def n_agents(self) -> int:
        return self.solar.shape[1]

    def hourly_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "step": np.arange(self.n_steps),
            "day": np.arange(self.n_steps) // 24,
            "hour": np.arange(self.n_steps) % 24,
            "solar_kwh": self.solar.sum(axis=1),
            "demand_kwh": self.demand.sum(axis=1),
            "grid_import_kwh": self.grid_import.sum(axis=1),
            "grid_export_kwh": self.grid_export.sum(axis=1),
            "p2p_traded_kwh": self.p2p_bought.sum(axis=1),
            "p2p_price_inr": self.p2p_price,
            "mean_soc_kwh": self.soc.mean(axis=1),
            "cost_inr": self.agent_cost.sum(axis=1),
        })

    def agent_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "agent_id": self.agent_ids,
            "tags": [",".join(t) for t in self.agent_tags],
            "cost_inr": self.agent_cost.sum(axis=0),
            "demand_kwh": self.demand.sum(axis=0),
            "solar_kwh": self.solar.sum(axis=0),
            "grid_import_kwh": self.grid_import.sum(axis=0),
            "grid_export_kwh": self.grid_export.sum(axis=0),
            "p2p_bought_kwh": self.p2p_bought.sum(axis=0),
            "p2p_sold_kwh": self.p2p_sold.sum(axis=0),
        })

"""
metrics.py - All evaluation metrics (PHASE 12). Pure functions of a SimulationResult.

Definitions (T hours, N households, 1 h steps so kWh == kW for a single hour):
  total_cost            = sum_{t,i} cost_{t,i}                      (INR, cash flows only)
  peak_grid_load        = max_t sum_i grid_import_{t,i}             (kW drawn from the main grid)
  grid_dependency       = total grid import / total demand
  renewable_utilization = 1 - total grid export / total solar  (solar kept in the neighbourhood,
                          incl. battery storage and local P2P; clipped to [0,1])
  local_energy_traded   = sum p2p_bought  (= sum p2p_sold)
  per-agent savings     = reference cost - cost, with two references:
                          (a) grid-only: buy all demand at the grid tariff (no PV/battery)
                          (b) baseline run
  jains_index           = (sum x)^2 / (n * sum x^2) on x_i = benefit ratio vs grid-only
  price_of_anarchy      = selfish total cost / cooperative total cost  (the PROJECT's simulation
                          comparison metric, NOT the formal worst-case game-theoretic PoA)
  soc_adjusted_cost     = total cost - value of net battery energy gained (fair when final SOCs differ)
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

_EPS = 1e-12


def total_cost(r) -> float:
    return float(np.sum(r.agent_cost))


def per_agent_cost(r) -> np.ndarray:
    return np.sum(r.agent_cost, axis=0)


def aggregate_grid_import(r) -> np.ndarray:
    return np.sum(r.grid_import, axis=1)


def peak_grid_load(r) -> float:
    return float(np.max(aggregate_grid_import(r)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > _EPS else 0.0


def grid_dependency(r) -> float:
    return _safe_ratio(np.sum(r.grid_import), np.sum(r.demand))


def renewable_utilization(r) -> float:
    solar = float(np.sum(r.solar))
    if solar <= _EPS:
        return 0.0
    return float(np.clip(1.0 - np.sum(r.grid_export) / solar, 0.0, 1.0))


def local_energy_traded(r) -> float:
    return float(np.sum(r.p2p_bought))


def grid_energy_purchased(r) -> float:
    return float(np.sum(r.grid_import))


def grid_energy_sold(r) -> float:
    return float(np.sum(r.grid_export))


def grid_only_reference_cost(r) -> np.ndarray:
    """Cost of each household if it had no PV, no battery and bought everything from the grid."""
    return np.sum(r.demand * r.grid_buy_price[:, None], axis=0)


def per_agent_savings_vs_grid_only(r) -> np.ndarray:
    return grid_only_reference_cost(r) - per_agent_cost(r)


def per_agent_savings_vs_baseline(r, baseline) -> np.ndarray:
    return per_agent_cost(baseline) - per_agent_cost(r)


def jains_index(values: Sequence[float]) -> float:
    """Jain's fairness index J = (sum x)^2 / (n sum x^2); 1/n <= J <= 1. All-zero -> 1.0."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("jains_index needs at least one value")
    denom = x.size * np.sum(x ** 2)
    if denom <= _EPS:
        return 1.0
    return float(np.sum(x) ** 2 / denom)


def benefit_ratios(r) -> np.ndarray:
    """x_i = relative saving of household i vs buying everything from the grid (clipped at 0)."""
    ref = grid_only_reference_cost(r)
    ratio = np.where(ref > _EPS, per_agent_savings_vs_grid_only(r) / np.where(ref > _EPS, ref, 1.0), 0.0)
    return np.clip(ratio, 0.0, None)


def fairness(r) -> float:
    return jains_index(benefit_ratios(r))


def soc_adjusted_cost(r) -> float:
    inventory = (r.final_soc - r.initial_soc) * r.inventory_value_per_kwh
    return total_cost(r) - float(np.sum(inventory))


def price_of_anarchy(selfish_cost: float, cooperative_cost: float) -> float:
    """Selfish total cost / cooperative total cost (NaN if the denominator is not positive)."""
    if cooperative_cost <= _EPS:
        return math.nan
    return float(selfish_cost / cooperative_cost)


def compute_all(r) -> dict:
    """The 11 required metrics (+ a few diagnostics) for one run."""
    return {
        "total_cost": total_cost(r),
        "soc_adjusted_cost": soc_adjusted_cost(r),
        "peak_grid_load": peak_grid_load(r),
        "grid_dependency": grid_dependency(r),
        "renewable_utilization": renewable_utilization(r),
        "local_energy_traded": local_energy_traded(r),
        "grid_energy_purchased": grid_energy_purchased(r),
        "grid_energy_sold": grid_energy_sold(r),
        "fairness_jain": fairness(r),
        "per_agent_cost": per_agent_cost(r).tolist(),
        "per_agent_savings_vs_grid_only": per_agent_savings_vs_grid_only(r).tolist(),
    }

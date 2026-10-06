"""
metrics.py - Evaluation metrics for one SimulationResult.

Core definitions:

    total_cost
        = actual household electricity cash cost

    peak_grid_load
        = maximum aggregate grid import in any hour

    peak_demand_charge
        = peak_grid_load * configured peak charge per kW

    system_cost
        = total_cost + peak_demand_charge

    grid_dependency
        = total grid import / total household demand

    renewable_utilization
        = 1 - total grid export / total solar generation

    local_energy_traded
        = total P2P energy bought (= total P2P energy sold)

    fairness
        = Jain's fairness index over per-agent benefits

    price_of_anarchy
        = selfish system cost / cooperative system cost

    soc_adjusted_cost
        = total cash cost - value of net battery inventory change

IMPORTANT:
    `total_cost` remains the actual electricity bill.

    `system_cost` is the neighbourhood-level social/system metric used
    to evaluate the trade-off between energy cost and peak grid demand.

    The project's PoA is based on system cost, not raw household cash cost.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


_EPS = 1e-12

# Fallback used for legacy SimulationResult objects/tests whose
# config dictionary does not contain the new field.
DEFAULT_PEAK_DEMAND_CHARGE_PER_KW = 100.0


# =====================================================================
# BASIC COST METRICS
# =====================================================================

def total_cost(r) -> float:
    """
    Total household electricity cash cost over the simulation.
    """
    return float(np.sum(r.agent_cost))


def per_agent_cost(r) -> np.ndarray:
    """
    Total electricity cash cost for each household.
    """
    return np.sum(r.agent_cost, axis=0)


# =====================================================================
# GRID METRICS
# =====================================================================

def aggregate_grid_import(r) -> np.ndarray:
    """
    Aggregate grid import across all households for every simulation hour.

    Returns:
        Array of shape (T,)
    """
    return np.sum(r.grid_import, axis=1)


def peak_grid_load(r) -> float:
    """
    Maximum aggregate energy drawn from the main grid in any hour.
    """
    return float(np.max(aggregate_grid_import(r)))


def _safe_ratio(num: float, den: float) -> float:
    """
    Safe division helper.
    """
    return float(num / den) if abs(den) > _EPS else 0.0


def grid_dependency(r) -> float:
    """
    Fraction of total household demand supplied by the main grid.
    """
    return _safe_ratio(
        float(np.sum(r.grid_import)),
        float(np.sum(r.demand))
    )


# =====================================================================
# COMMUNITY / SYSTEM COST
# =====================================================================

def peak_demand_charge(
    r,
    peak_charge_per_kw: Optional[float] = None
) -> float:
    """
    Calculate the neighbourhood-level peak demand charge.

    Formula:

        Peak Demand Charge
            = Peak Grid Load × charge per kW

    The explicit argument takes priority.

    Otherwise we try to read:
        r.config["peak_demand_charge_per_kw"]

    Legacy results without this field use the project default.
    """

    if peak_charge_per_kw is None:

        peak_charge_per_kw = float(
            r.config.get(
                "peak_demand_charge_per_kw",
                DEFAULT_PEAK_DEMAND_CHARGE_PER_KW
            )
        )

    if peak_charge_per_kw < 0:
        raise ValueError(
            "peak_charge_per_kw must be >= 0"
        )

    return (
        peak_grid_load(r)
        * peak_charge_per_kw
    )


def system_cost(
    r,
    peak_charge_per_kw: Optional[float] = None
) -> float:
    """
    Neighbourhood-level system/social cost.

    Formula:

        System Cost
            = Energy Cost
            + Peak Demand Charge
    """

    return (
        total_cost(r)
        + peak_demand_charge(
            r,
            peak_charge_per_kw
        )
    )


# =====================================================================
# RENEWABLE / TRADING METRICS
# =====================================================================

def renewable_utilization(r) -> float:
    """
    Fraction of generated solar energy retained within the neighbourhood.

    This accounts for energy consumed locally, stored in batteries and
    exchanged through P2P trading.

    Equivalent implementation:
        1 - grid_export / solar_generation

    Result is clipped to [0, 1].
    """

    solar = float(
        np.sum(r.solar)
    )

    if solar <= _EPS:
        return 0.0

    utilization = (
        1.0
        - float(np.sum(r.grid_export)) / solar
    )

    return float(
        np.clip(
            utilization,
            0.0,
            1.0
        )
    )


def local_energy_traded(r) -> float:
    """
    Total energy traded through the local P2P market.
    """

    return float(
        np.sum(r.p2p_bought)
    )


def grid_energy_purchased(r) -> float:
    """
    Total energy purchased from the main grid.
    """

    return float(
        np.sum(r.grid_import)
    )


def grid_energy_sold(r) -> float:
    """
    Total energy exported to the main grid.
    """

    return float(
        np.sum(r.grid_export)
    )


# =====================================================================
# REFERENCE COST / SAVINGS
# =====================================================================

def grid_only_reference_cost(r) -> np.ndarray:
    """
    Cost for each household if it had:

        - no solar
        - no battery
        - no P2P trading

    and bought all demand directly from the main grid.
    """

    return np.sum(
        r.demand
        * r.grid_buy_price[:, None],
        axis=0
    )


def per_agent_savings_vs_grid_only(r) -> np.ndarray:
    """
    Savings of each household against the grid-only reference.
    """

    return (
        grid_only_reference_cost(r)
        - per_agent_cost(r)
    )


def per_agent_savings_vs_baseline(
    r,
    baseline
) -> np.ndarray:
    """
    Savings relative to the baseline scenario.
    """

    return (
        per_agent_cost(baseline)
        - per_agent_cost(r)
    )


# =====================================================================
# FAIRNESS
# =====================================================================

def jains_index(
    values: Sequence[float]
) -> float:
    """
    Jain's fairness index:

        J = (sum x)^2 / (n * sum(x^2))

    Range:
        1/n <= J <= 1

    All-zero values -> 1.0
    """

    x = np.asarray(
        values,
        dtype=float
    )

    if x.size == 0:
        raise ValueError(
            "jains_index needs at least one value"
        )

    denominator = (
        x.size
        * np.sum(x ** 2)
    )

    if denominator <= _EPS:
        return 1.0

    return float(
        np.sum(x) ** 2
        / denominator
    )


def benefit_ratios(r) -> np.ndarray:
    """
    Relative benefit of each household versus grid-only operation.

    Negative savings are clipped to zero because this metric measures
    realised benefit rather than penalty.
    """

    reference = (
        grid_only_reference_cost(r)
    )

    savings = (
        per_agent_savings_vs_grid_only(r)
    )

    ratio = np.where(
        reference > _EPS,
        savings / np.where(
            reference > _EPS,
            reference,
            1.0
        ),
        0.0
    )

    return np.clip(
        ratio,
        0.0,
        None
    )


def fairness(r) -> float:
    """
    Jain's fairness index over household benefit ratios.
    """

    return jains_index(
        benefit_ratios(r)
    )


# =====================================================================
# SOC-ADJUSTED COST
# =====================================================================

def soc_adjusted_cost(r) -> float:
    """
    Adjust cash cost for changes in battery inventory.

    If final SOC is higher than initial SOC, the additional stored
    energy has value and is subtracted from cash cost.

    This prevents unfair comparison caused only by different ending SOC.
    """

    inventory = (
        r.final_soc
        - r.initial_soc
    ) * r.inventory_value_per_kwh

    return (
        total_cost(r)
        - float(np.sum(inventory))
    )


# =====================================================================
# PRICE OF ANARCHY
# =====================================================================

def price_of_anarchy(
    selfish_cost: float,
    cooperative_cost: float
) -> float:
    """
    Project-level simulation comparison metric.

    PoA = Selfish System Cost / Cooperative System Cost

    Interpretation:

        PoA > 1
            selfish behaviour has higher system cost

        PoA = 1
            both strategies have equal system cost

        PoA < 1
            selfish strategy happened to have lower system cost

    This is a project comparison metric, not the formal worst-case
    game-theoretic Price of Anarchy.
    """

    if cooperative_cost <= _EPS:
        return math.nan

    return float(
        selfish_cost
        / cooperative_cost
    )


# =====================================================================
# ALL METRICS
# =====================================================================

def compute_all(
    r,
    peak_charge_per_kw: Optional[float] = None
) -> dict:
    """
    Compute all evaluation metrics for one simulation run.

    `peak_charge_per_kw` is optional so that existing unit tests and
    previously generated SimulationResult objects remain compatible.
    """

    return {

        # -------------------------------------------------------------
        # Cost
        # -------------------------------------------------------------

        "total_cost":
            total_cost(r),

        "soc_adjusted_cost":
            soc_adjusted_cost(r),

        "peak_demand_charge":
            peak_demand_charge(
                r,
                peak_charge_per_kw
            ),

        "system_cost":
            system_cost(
                r,
                peak_charge_per_kw
            ),

        # -------------------------------------------------------------
        # Grid / energy
        # -------------------------------------------------------------

        "peak_grid_load":
            peak_grid_load(r),

        "grid_dependency":
            grid_dependency(r),

        "renewable_utilization":
            renewable_utilization(r),

        "local_energy_traded":
            local_energy_traded(r),

        "grid_energy_purchased":
            grid_energy_purchased(r),

        "grid_energy_sold":
            grid_energy_sold(r),

        # -------------------------------------------------------------
        # Fairness
        # -------------------------------------------------------------

        "fairness_jain":
            fairness(r),

        # -------------------------------------------------------------
        # Per-agent diagnostics
        # -------------------------------------------------------------

        "per_agent_cost":
            per_agent_cost(r).tolist(),

        "per_agent_savings_vs_grid_only":
            per_agent_savings_vs_grid_only(r).tolist(),
    }
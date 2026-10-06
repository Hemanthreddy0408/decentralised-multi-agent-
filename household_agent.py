"""
household_agent.py - Autonomous household agent (PHASES 3, 4).

STATE  S = (hour, solar, demand, battery_soc, grid_price, surplus_or_deficit, public_info)
   locally observable : hour, own solar, own demand, own SOC, own surplus/deficit
   publicly broadcast : grid prices, surplus/deficit messages, past clearing price,
                        aggregate neighbourhood load (``PublicInfo``)
   private            : own future demand/solar forecast, SA state & plan, bargaining shade,
                        internal cost (never visible to other agents)
ACTIONS  charge / discharge / hold (from SA plan), then sell / buy (auction + grid).
TRANSITION (one hour):
   net = solar - demand - charge_drawn + discharge_delivered
   net > 0 -> surplus (sell),  net < 0 -> deficit (buy)
   soc' = soc + charge_drawn*eta_c - discharge_delivered/eta_d
ENERGY BALANCE (checked every hour):
   solar + grid_import + p2p_bought + discharge = demand + charge + grid_export + p2p_sold
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from battery import Battery
from communication import BroadcastMessage, PublicInfo
from config import Config, HOURS_PER_DAY
from data_generator import HouseholdProfile
from double_auction import Bid
from simulated_annealing import LEVELS, HOLD, SAParams, SAResult, optimize_schedule
from strategies import Strategy

_EPS = 1e-9


class HouseholdAgent:
    def __init__(self, profile: HouseholdProfile, cfg: Config, strategy: Strategy, seed: int):
        self.agent_id = profile.household_id
        self.tags = list(profile.tags)
        self.battery = Battery(profile.battery_capacity, profile.initial_soc, profile.max_charge_rate,
                               profile.max_discharge_rate, profile.charge_efficiency,
                               profile.discharge_efficiency, profile.min_soc_fraction)
        self.strategy = strategy
        self.cfg = cfg
        self.sa_params = SAParams.from_config(cfg)
        self._rng = random.Random(seed * 100003 + self.agent_id)    # private SA randomness
        # --- private knowledge: the typical day the household has learned over time
        self._solar_base = np.asarray(profile.solar_base, float)
        self._demand_base = np.asarray(profile.demand_base, float)
        self._solar_scale = 1.0          # learned "today is cloudier/sunnier than usual"
        self._demand_scale = 1.0
        self._plan: Optional[list] = None
        # --- local observation
        self.hour = 0
        self.solar = 0.0
        self.demand = 0.0
        # --- last-step physical flows
        self.charge_in = self.discharge_out = 0.0
        self.surplus = self.deficit = 0.0
        self.grid_import = self.grid_export = 0.0
        self.p2p_bought = self.p2p_sold = 0.0
        # --- accounting
        self.internal_cost = 0.0
        self.energy_bought_grid = self.energy_sold_grid = 0.0
        self.energy_traded_locally = 0.0
        self.sa_calls = self.sa_iterations = 0
        self.sa_improvement = 0.0

    # ----------------------------------------------------- observe / forecast
    def sense(self, hour: int, solar: float, demand: float) -> None:
        """Observe own solar & demand for this hour and update the private forecast scales."""
        if solar < 0 or demand < 0:
            raise ValueError("solar and demand must be non-negative")
        self.hour, self.solar, self.demand = hour, solar, demand
        if self._solar_base[hour] > 0.05:                         # daylight: learn cloud factor
            ratio = solar / self._solar_base[hour]
            self._solar_scale = float(np.clip(0.5 * self._solar_scale + 0.5 * ratio, 0.2, 1.5))
        if self._demand_base[hour] > 0.05:
            ratio = demand / self._demand_base[hour]
            self._demand_scale = float(np.clip(0.8 * self._demand_scale + 0.2 * ratio, 0.5, 1.5))

    def forecast_net_load(self, horizon: int) -> np.ndarray:
        """Forecast (demand - solar) for the next ``horizon`` hours; hour 0 is observed exactly."""
        out = np.empty(horizon)
        out[0] = self.demand - self.solar
        for k in range(1, horizon):
            h = (self.hour + k) % HOURS_PER_DAY
            out[k] = self._demand_base[h] * self._demand_scale - self._solar_base[h] * self._solar_scale
        return out

    # ------------------------------------------------------------- planning
    def plan(self, public: PublicInfo) -> SAResult:
        """Replan the whole 24-h schedule with Simulated Annealing (every hour)."""
        problem = self.strategy.build_problem(self, public)
        result = optimize_schedule(problem, self._rng, self.sa_params, warm_start=self._plan)
        self._plan = result.best_schedule
        self.sa_calls += 1
        self.sa_iterations += result.iterations
        self.sa_improvement += result.initial_cost - result.best_cost
        return result

    def execute(self) -> None:
        """Execute ONLY the first action of the plan, then compute surplus/deficit."""
        level = LEVELS[self._plan[0]] if self._plan else 0.0
        b = self.battery
        self.charge_in, self.discharge_out = b.step(
            charge_request=max(level, 0.0) * b.max_charge_rate,
            discharge_request=max(-level, 0.0) * b.max_discharge_rate)
        net = self.solar - self.demand - self.charge_in + self.discharge_out
        self.surplus = max(net, 0.0)
        self.deficit = max(-net, 0.0)
        self.p2p_bought = self.p2p_sold = 0.0

    # ---------------------------------------------------------- communication
    def broadcast(self) -> Optional[BroadcastMessage]:
        if self.surplus > _EPS:
            return BroadcastMessage(self.agent_id, self.hour, "surplus", self.surplus)
        if self.deficit > _EPS:
            return BroadcastMessage(self.agent_id, self.hour, "deficit", self.deficit)
        return None

    def make_bid(self, public: PublicInfo) -> Optional[Bid]:
        return self.strategy.make_bid(self, public) if self.strategy.uses_auction else None

    # ------------------------------------------------------------ settlement
    def settle(self, p2p_bought: float, p2p_sold: float, p2p_price: float,
               grid_buy_price: float, grid_sell_price: float) -> None:
        """Apply auction outcome, send the remainder to the grid, update cost & accounting."""
        if p2p_bought > self.deficit + 1e-6 or p2p_sold > self.surplus + 1e-6:
            raise ValueError("P2P quantity exceeds the household's deficit/surplus")
        self.p2p_bought, self.p2p_sold = p2p_bought, p2p_sold
        self.grid_import = max(self.deficit - p2p_bought, 0.0)       # unmatched deficit -> grid
        self.grid_export = max(self.surplus - p2p_sold, 0.0)         # unmatched surplus -> grid
        cost = (self.grid_import * grid_buy_price - self.grid_export * grid_sell_price
                + (p2p_bought - p2p_sold) * (p2p_price if p2p_price is not None else 0.0))
        self.internal_cost += cost
        self.energy_bought_grid += self.grid_import
        self.energy_sold_grid += self.grid_export
        self.energy_traded_locally += p2p_bought + p2p_sold
        self._last_cost = cost

    def energy_balance_error(self) -> float:
        """Should be ~0: supply side minus demand side of this hour's energy balance."""
        supply = self.solar + self.grid_import + self.p2p_bought + self.discharge_out
        use = self.demand + self.charge_in + self.grid_export + self.p2p_sold
        return supply - use

    @property
    def last_cost(self) -> float:
        return getattr(self, "_last_cost", 0.0)

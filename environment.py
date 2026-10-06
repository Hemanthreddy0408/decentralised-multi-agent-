"""
environment.py - SmartGridEnvironment (PHASES 7, 11).

The environment is a *simulator*, not a decision maker.  It
  * advances time and feeds every agent ONLY its own sensor readings,
  * relays the public broadcast messages and runs the Double Auction (the market),
  * settles leftovers with the main grid, checks energy balance, and records history.
It never chooses a battery action, a price or a quantity for any household.

Hourly loop (matches the project specification):
  1 observe   2 update public info   3 replan with SA   4 execute battery action
  5 surplus/deficit   6 broadcast   7 bids   8 Double Auction   9 execute trades
  10 unmatched -> grid   11 battery update   12 costs   13 record   14 next hour
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from communication import PublicInfo
from config import Config, HOURS_PER_DAY, SCENARIOS
from data_generator import HouseholdProfile
from double_auction import run_double_auction
from household_agent import HouseholdAgent
from results import SimulationResult
from strategies import make_strategy

_BALANCE_TOL = 1e-6


class SmartGridEnvironment:
    def __init__(self, cfg: Config, profiles: list[HouseholdProfile], scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if not profiles:
            raise ValueError("at least one household profile is required")
        self.cfg = cfg
        self.scenario = scenario
        self.profiles = profiles
        self.n_agents = len(profiles)
        self.n_steps = cfg.n_days * HOURS_PER_DAY
        for p in profiles:
            if len(p.solar_actual) < self.n_steps or len(p.demand_actual) < self.n_steps:
                raise ValueError(f"profile of household {p.household_id} is shorter than the simulation")
        self.grid_buy = cfg.grid_buy_prices()
        self.grid_sell = cfg.grid_sell_prices()
        self.public = PublicInfo.initial(self.grid_buy, self.grid_sell)
        self.agents = [HouseholdAgent(p, cfg, make_strategy(scenario, cfg, cfg.seed + 7919 * (p.household_id + 1)),
                                      cfg.seed) for p in profiles]
        self._alloc_history()
        self.t = 0

    # ------------------------------------------------------------------ history
    def _alloc_history(self) -> None:
        shape = (self.n_steps, self.n_agents)
        names = ["solar", "demand", "soc", "charge_in", "discharge_out", "grid_import",
                 "grid_export", "p2p_bought", "p2p_sold", "agent_cost"]
        self.hist = {n: np.zeros(shape) for n in names}
        self.hist["p2p_price"] = np.full(self.n_steps, np.nan)

    # --------------------------------------------------------------- one hour
    def step(self) -> None:
        t = self.t
        hour = t % HOURS_PER_DAY
        self.public.hour = hour
        buy, sell = self.grid_buy[hour], self.grid_sell[hour]
        # 1-2 observe local state (each agent receives only ITS OWN readings)
        for a, p in zip(self.agents, self.profiles):
            a.sense(hour, float(p.solar_actual[t]), float(p.demand_actual[t]))
        # 3-4 replan with Simulated Annealing, execute the first action
        for a in self.agents:
            a.plan(self.public)
            a.execute()
        # 5-6 broadcast surplus / deficit
        self.public.broadcasts = [m for m in (a.broadcast() for a in self.agents) if m is not None]
        # 7-8 bids + Double Auction (baseline has no local market)
        bids = [b for b in (a.make_bid(self.public) for a in self.agents) if b is not None]
        auction = run_double_auction(bids, self.cfg.auction_k)
        # 9-12 execute trades, unmatched energy to the grid, update costs
        for a in self.agents:
            a.settle(auction.bought_by(a.agent_id), auction.sold_by(a.agent_id),
                     auction.clearing_price, buy, sell)
            err = a.energy_balance_error()
            if abs(err) > _BALANCE_TOL:
                raise AssertionError(f"energy balance violated for agent {a.agent_id} at t={t}: {err}")
        # 13 record
        h = self.hist
        for i, a in enumerate(self.agents):
            h["solar"][t, i], h["demand"][t, i] = a.solar, a.demand
            h["soc"][t, i] = a.battery.soc
            h["charge_in"][t, i], h["discharge_out"][t, i] = a.charge_in, a.discharge_out
            h["grid_import"][t, i], h["grid_export"][t, i] = a.grid_import, a.grid_export
            h["p2p_bought"][t, i], h["p2p_sold"][t, i] = a.p2p_bought, a.p2p_sold
            h["agent_cost"][t, i] = a.last_cost
        if auction.clearing_price is not None:
            h["p2p_price"][t] = auction.clearing_price
            self.public.clearing_ema[hour] = 0.7 * self.public.clearing_ema[hour] + 0.3 * auction.clearing_price
        # public aggregate (transformer meter) load, smoothed per hour of day
        load = float(h["grid_import"][t].sum())
        if self.public.load_seen[hour]:
            self.public.agg_load_ema[hour] = 0.5 * self.public.agg_load_ema[hour] + 0.5 * load
        else:
            self.public.agg_load_ema[hour], self.public.load_seen[hour] = load, True
        self.t += 1                                           # 14 next hour

    # --------------------------------------------------------------------- run
    def run(self, verbose: bool = False) -> SimulationResult:
        start = time.time()
        while self.t < self.n_steps:
            self.step()
            if verbose and self.t % HOURS_PER_DAY == 0:
                print(f"  [{self.scenario}] day {self.t // HOURS_PER_DAY}/{self.cfg.n_days} done "
                      f"({time.time() - start:.1f}s)")
        return self.result()

    def result(self) -> SimulationResult:
        h = self.hist
        mean_buy = float(self.grid_buy.mean())
        calls = sum(a.sa_calls for a in self.agents)
        return SimulationResult(
            scenario=self.scenario, config=self.cfg.to_dict(),
            agent_ids=[a.agent_id for a in self.agents], agent_tags=[a.tags for a in self.agents],
            solar=h["solar"], demand=h["demand"], soc=h["soc"], charge_in=h["charge_in"],
            discharge_out=h["discharge_out"], grid_import=h["grid_import"], grid_export=h["grid_export"],
            p2p_bought=h["p2p_bought"], p2p_sold=h["p2p_sold"], agent_cost=h["agent_cost"],
            grid_buy_price=np.tile(self.grid_buy, self.cfg.n_days),
            grid_sell_price=np.tile(self.grid_sell, self.cfg.n_days), p2p_price=h["p2p_price"],
            initial_soc=np.array([p.initial_soc for p in self.profiles]),
            final_soc=np.array([a.battery.soc for a in self.agents]),
            inventory_value_per_kwh=np.array([
                self.cfg.terminal_value_factor * a.battery.discharge_efficiency * mean_buy for a in self.agents]),
            sa_stats={"sa_calls": calls,
                      "sa_iterations_total": int(sum(a.sa_iterations for a in self.agents)),
                      "sa_mean_iterations": sum(a.sa_iterations for a in self.agents) / max(calls, 1),
                      "sa_mean_improvement_inr": sum(a.sa_improvement for a in self.agents) / max(calls, 1)})

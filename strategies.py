"""
strategies.py - The three experimental configurations (PHASES 8-10).

A Strategy is owned by ONE agent; it turns the agent's own observations plus the
broadcast ``PublicInfo`` into (a) a PlanningProblem for Simulated Annealing and
(b) a bid for the auction.  No strategy can see other agents' private state.

BASELINE     : cost objective on plain grid prices, no auction (grid-only trading).
SELFISH      : minimise OWN cost; values exports/imports with the public P2P price signal;
               bids strategically (shades its limit price towards its own advantage,
               using the broadcast supply/demand ratio as bargaining power).
COOPERATIVE  : same own-cost objective PLUS a community term built only from public
               aggregate information:
                 * congestion surcharge  c_t = w_c * p_ref * max(f_t - 1, 0)  added to the price
                   of every kWh IMPORTED, f_t = public forecast of neighbourhood load / its mean.
                   (importing when the neighbourhood is congested costs extra, so agents use
                   their battery / shift load away from the community peak)
                 * quadratic peak penalty  w_p * f_t * grid_import^2
               and bids truthfully (ask = grid sell price, bid = grid buy price) so that the
               maximum amount of energy stays inside the neighbourhood.
               Each agent still optimises ITS OWN schedule: there is no central optimiser.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config import Config, HOURS_PER_DAY
from communication import PublicInfo
from double_auction import Bid
from simulated_annealing import PlanningProblem

_EPS = 1e-9


def _window(array24: np.ndarray, hour: int, horizon: int) -> np.ndarray:
    """Values of a 24-h periodic array for the next ``horizon`` hours, starting at ``hour``."""
    return np.array([array24[(hour + k) % HOURS_PER_DAY] for k in range(horizon)])


class Strategy(ABC):
    name = "abstract"
    uses_auction = False

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ planning
    def _problem(self, agent, public: PublicInfo, buy_plan, sell_plan, congestion=None, peak_weight=None):
        cfg, H = self.cfg, self.cfg.horizon
        b = agent.battery
        mean_buy = float(np.mean(buy_plan))
        return PlanningProblem(
            net_load=agent.forecast_net_load(H).tolist(),
            buy_price=list(buy_plan), sell_price=list(sell_plan),
            soc0=b.soc, capacity=b.capacity, min_soc=b.min_soc,
            max_charge=b.max_charge_rate, max_discharge=b.max_discharge_rate,
            charge_eff=b.charge_efficiency, discharge_eff=b.discharge_efficiency,
            degradation_cost=cfg.degradation_cost,
            terminal_value=cfg.terminal_value_factor * b.discharge_efficiency * mean_buy,
            congestion=None if congestion is None else list(congestion),
            peak_weight=None if peak_weight is None else list(peak_weight),
            violation_penalty=cfg.violation_penalty)

    @abstractmethod
    def build_problem(self, agent, public: PublicInfo) -> PlanningProblem: ...

    # ------------------------------------------------------------------- bidding
    def make_bid(self, agent, public: PublicInfo) -> Optional[Bid]:
        return None

    def _bid_with_shade(self, agent, public, shade_sell: float, shade_buy: float) -> Optional[Bid]:
        buy, sell = public.grid_buy[agent.hour], public.grid_sell[agent.hour]
        spread = buy - sell
        if agent.surplus > _EPS:
            return Bid(agent.agent_id, "sell", agent.surplus, sell + shade_sell * spread)
        if agent.deficit > _EPS:
            return Bid(agent.agent_id, "buy", agent.deficit, buy - shade_buy * spread)
        return None


class BaselineStrategy(Strategy):
    """No neighbourhood trading: plan on grid prices only; surplus/deficit go to the grid."""
    name = "baseline"
    uses_auction = False

    def build_problem(self, agent, public):
        H = self.cfg.horizon
        return self._problem(agent, public, _window(public.grid_buy, agent.hour, H),
                             _window(public.grid_sell, agent.hour, H))


class SelfishStrategy(Strategy):
    """Rational, self-interested agent: minimises own cost, bargains strategically."""
    name = "selfish"
    uses_auction = True

    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__(cfg, seed)
        lo, hi = cfg.selfish_shade_range
        self.base_shade = self.rng.uniform(lo, hi)      # PRIVATE bargaining aggressiveness

    def _blended_prices(self, public, hour):
        H, kappa = self.cfg.horizon, self.cfg.price_blend_kappa
        buy, sell = _window(public.grid_buy, hour, H), _window(public.grid_sell, hour, H)
        clear = _window(public.clearing_ema, hour, H)
        # expected effective prices: mix of grid tariff and the public P2P clearing price
        return buy - kappa * (buy - clear), sell + kappa * (clear - sell)

    def build_problem(self, agent, public):
        buy_plan, sell_plan = self._blended_prices(public, agent.hour)
        return self._problem(agent, public, buy_plan, sell_plan)

    def make_bid(self, agent, public):
        # Bargaining power from PUBLIC broadcasts: scarce supply -> sellers push their ask
        # up, buyers shade less; abundant supply -> the opposite.
        supply, demand = public.total_surplus(), public.total_deficit()
        ratio = (demand + 1e-6) / (supply + 1e-6)
        shade_sell = self.base_shade * float(np.clip(ratio, 0.6, 1.4))
        shade_buy = self.base_shade * float(np.clip(1.0 / ratio, 0.6, 1.4))
        return self._bid_with_shade(agent, public, shade_sell, shade_buy)


class CooperativeStrategy(SelfishStrategy):
    """Own cost + public community terms (congestion, peak); truthful bidding."""
    name = "cooperative"
    uses_auction = True

    def build_problem(self, agent, public):
        cfg, H = self.cfg, self.cfg.horizon
        buy_plan, sell_plan = self._blended_prices(public, agent.hour)
        f = public.congestion_factor(agent.hour, H)               # PUBLIC community signal
        p_ref = float(np.mean(_window(public.grid_buy, agent.hour, H)))
        congestion = cfg.coop_congestion_weight * p_ref * np.clip(f - 1.0, 0.0, None)
        peak_weight = cfg.coop_peak_weight * f
        return self._problem(agent, public, buy_plan, sell_plan, congestion, peak_weight)

    def make_bid(self, agent, public):
        return self._bid_with_shade(agent, public, 0.0, 0.0)      # truthful: ask=sell, bid=buy


STRATEGIES = {"baseline": BaselineStrategy, "selfish": SelfishStrategy, "cooperative": CooperativeStrategy}


def make_strategy(name: str, cfg: Config, seed: int) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}', choose from {list(STRATEGIES)}")
    return STRATEGIES[name](cfg, seed)

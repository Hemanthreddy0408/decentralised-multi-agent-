"""
strategies.py - Experimental agent strategies.

The project contains three configurations:

1. BASELINE
   No peer-to-peer market.
   Households use only the main grid.

2. SELFISH
   Each household independently minimises its own electricity cost.
   Agents use the public P2P price signal and strategically shade bids.

3. COOPERATIVE
   Each household still independently optimises its own schedule,
   but its objective is augmented with PUBLIC neighbourhood signals:

       - congestion pressure
       - aggregate load pressure
       - peak-import penalty

   No household receives another household's private battery SOC,
   private valuation, future demand, or Simulated Annealing state.

FOAI concepts:
- autonomous agents
- rational/selfish agents
- cooperative agents
- partial observability
- dynamic environment
- informed search
- decentralized communication
- negotiation
- multi-agent interaction
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


# =====================================================================
# HELPER
# =====================================================================

def _window(
    array24: np.ndarray,
    hour: int,
    horizon: int
) -> np.ndarray:
    """
    Extract the next `horizon` values from a 24-hour periodic array.

    Example:
        hour = 22
        horizon = 5

    returns:
        [22, 23, 0, 1, 2]
    """

    return np.array(
        [
            array24[(hour + k) % HOURS_PER_DAY]
            for k in range(horizon)
        ],
        dtype=float
    )


# =====================================================================
# BASE STRATEGY
# =====================================================================

class Strategy(ABC):

    name = "abstract"

    uses_auction = False

    def __init__(
        self,
        cfg: Config,
        seed: int = 0
    ):
        self.cfg = cfg

        # Private random generator for this strategy.
        self.rng = random.Random(seed)

    # -----------------------------------------------------------------
    # Planning problem
    # -----------------------------------------------------------------

    def _problem(
        self,
        agent,
        public: PublicInfo,
        buy_plan,
        sell_plan,
        congestion=None,
        peak_weight=None
    ) -> PlanningProblem:
        """
        Construct the local PlanningProblem passed to Simulated Annealing.

        The important point is that every strategy creates a planning
        problem for ONE household only.

        Other agents' private state is NOT passed here.
        """

        cfg = self.cfg

        H = cfg.horizon

        battery = agent.battery

        mean_buy_price = float(
            np.mean(buy_plan)
        )

        return PlanningProblem(

            # Household's forecasted net load:
            # positive -> deficit
            # negative -> surplus
            net_load=(
                agent
                .forecast_net_load(H)
                .tolist()
            ),

            buy_price=list(buy_plan),
            sell_price=list(sell_plan),

            # Current battery state
            soc0=battery.soc,

            capacity=battery.capacity,
            min_soc=battery.min_soc,

            max_charge=battery.max_charge_rate,
            max_discharge=battery.max_discharge_rate,

            charge_eff=battery.charge_efficiency,
            discharge_eff=battery.discharge_efficiency,

            # Battery degradation
            degradation_cost=(
                cfg.degradation_cost
            ),

            # Value remaining energy at end of horizon
            terminal_value=(
                cfg.terminal_value_factor
                * battery.discharge_efficiency
                * mean_buy_price
            ),

            # Cooperative public signals
            congestion=(
                None
                if congestion is None
                else list(congestion)
            ),

            peak_weight=(
                None
                if peak_weight is None
                else list(peak_weight)
            ),

            # Strong physical constraint penalties
            violation_penalty=(
                cfg.violation_penalty
            ),
        )

    @abstractmethod
    def build_problem(
        self,
        agent,
        public: PublicInfo
    ) -> PlanningProblem:
        """
        Create the planning problem for the household.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------
    # Bidding
    # -----------------------------------------------------------------

    def make_bid(
        self,
        agent,
        public: PublicInfo
    ) -> Optional[Bid]:
        """
        Baseline does not participate in the P2P auction.
        """

        return None

    def _bid_with_shade(
        self,
        agent,
        public: PublicInfo,
        shade_sell: float,
        shade_buy: float
    ) -> Optional[Bid]:
        """
        Create a buy or sell bid.

        Seller:
            ask = grid_sell + seller_shading

        Buyer:
            bid = grid_buy - buyer_shading

        A cooperative agent can call this method with
        shade_sell = shade_buy = 0 to submit truthful bids.
        """

        hour = agent.hour

        grid_buy = float(
            public.grid_buy[hour]
        )

        grid_sell = float(
            public.grid_sell[hour]
        )

        spread = grid_buy - grid_sell

        # -------------------------------------------------------------
        # Seller
        # -------------------------------------------------------------

        if agent.surplus > _EPS:

            ask_price = (
                grid_sell
                + shade_sell * spread
            )

            return Bid(
                agent.agent_id,
                "sell",
                float(agent.surplus),
                float(ask_price)
            )

        # -------------------------------------------------------------
        # Buyer
        # -------------------------------------------------------------

        if agent.deficit > _EPS:

            bid_price = (
                grid_buy
                - shade_buy * spread
            )

            return Bid(
                agent.agent_id,
                "buy",
                float(agent.deficit),
                float(bid_price)
            )

        return None


# =====================================================================
# BASELINE
# =====================================================================

class BaselineStrategy(Strategy):
    """
    No local neighbourhood trading.

    The household plans against grid prices and eventually buys/sells
    unmatched energy directly with the main grid.
    """

    name = "baseline"

    uses_auction = False

    def build_problem(
        self,
        agent,
        public
    ):

        H = self.cfg.horizon

        buy_plan = _window(
            public.grid_buy,
            agent.hour,
            H
        )

        sell_plan = _window(
            public.grid_sell,
            agent.hour,
            H
        )

        return self._problem(
            agent,
            public,
            buy_plan,
            sell_plan
        )


# =====================================================================
# SELFISH
# =====================================================================

class SelfishStrategy(Strategy):
    """
    Rational self-interested household.

    Objective:
        minimise own electricity cost.

    The agent may use public P2P clearing-price information but does
    not receive private information belonging to other households.
    """

    name = "selfish"

    uses_auction = True

    def __init__(
        self,
        cfg: Config,
        seed: int = 0
    ):

        super().__init__(
            cfg,
            seed
        )

        low, high = (
            cfg.selfish_shade_range
        )

        # PRIVATE bargaining characteristic.
        self.base_shade = self.rng.uniform(
            low,
            high
        )

    # -----------------------------------------------------------------
    # Public P2P price blending
    # -----------------------------------------------------------------

    def _blended_prices(
        self,
        public: PublicInfo,
        hour: int
    ):
        """
        Estimate future effective P2P prices.

        Public market information is blended with grid tariffs.
        """

        H = self.cfg.horizon

        kappa = (
            self.cfg.price_blend_kappa
        )

        buy = _window(
            public.grid_buy,
            hour,
            H
        )

        sell = _window(
            public.grid_sell,
            hour,
            H
        )

        clearing = _window(
            public.clearing_ema,
            hour,
            H
        )

        # Effective buyer price:
        # closer to grid price when kappa is low,
        # closer to public P2P price when kappa is high.
        effective_buy = (
            buy
            - kappa * (buy - clearing)
        )

        # Effective seller price:
        effective_sell = (
            sell
            + kappa * (clearing - sell)
        )

        return (
            effective_buy,
            effective_sell
        )

    # -----------------------------------------------------------------
    # Selfish planning
    # -----------------------------------------------------------------

    def build_problem(
        self,
        agent,
        public: PublicInfo
    ):

        buy_plan, sell_plan = (
            self._blended_prices(
                public,
                agent.hour
            )
        )

        return self._problem(
            agent,
            public,
            buy_plan,
            sell_plan
        )

    # -----------------------------------------------------------------
    # Selfish bidding
    # -----------------------------------------------------------------

    def make_bid(
        self,
        agent,
        public: PublicInfo
    ):
        """
        Strategically shaded bidding.

        Supply and demand are PUBLIC values.

        High demand / scarce supply:
            sellers become more aggressive,
            buyers shade less.

        Low demand / abundant supply:
            more competitive prices.
        """

        public_supply = (
            public.total_surplus()
        )

        public_demand = (
            public.total_deficit()
        )

        ratio = (
            public_demand + 1e-6
        ) / (
            public_supply + 1e-6
        )

        ratio = float(
            np.clip(
                ratio,
                0.6,
                1.4
            )
        )

        shade_sell = (
            self.base_shade
            * ratio
        )

        shade_buy = (
            self.base_shade
            * (1.0 / ratio)
        )

        return self._bid_with_shade(
            agent,
            public,
            shade_sell,
            shade_buy
        )


# =====================================================================
# COOPERATIVE
# =====================================================================

class CooperativeStrategy(SelfishStrategy):
    """
    Cooperative household strategy.

    The household still independently optimises its own battery
    schedule.

    The difference is that its objective incorporates PUBLIC
    neighbourhood-level information.

    Public signals used:

        1. congestion factor
        2. historical aggregate grid load
        3. public P2P clearing prices

    Private information from other households is NOT used.

    The cooperative objective therefore becomes approximately:

        Own Energy Cost
        + Congestion Cost
        + Peak/Grid-Import Pressure

    This encourages agents to:
        - avoid high-load periods
        - use their batteries more intelligently
        - reduce unnecessary grid imports
        - retain more energy locally
    """

    name = "cooperative"

    uses_auction = True

    # -----------------------------------------------------------------
    # Cooperative planning
    # -----------------------------------------------------------------

    def build_problem(
        self,
        agent,
        public: PublicInfo
    ):

        cfg = self.cfg

        H = cfg.horizon

        # -------------------------------------------------------------
        # Base P2P-aware prices
        # -------------------------------------------------------------

        buy_plan, sell_plan = (
            self._blended_prices(
                public,
                agent.hour
            )
        )

        # -------------------------------------------------------------
        # PUBLIC congestion signal
        # -------------------------------------------------------------

        congestion_factor = (
            public.congestion_factor(
                agent.hour,
                H
            )
        )

        # -------------------------------------------------------------
        # PUBLIC historical aggregate load
        # -------------------------------------------------------------

        if np.any(public.load_seen):

            public_mean_load = float(
                np.mean(
                    public.agg_load_ema[
                        public.load_seen
                    ]
                )
            )

        else:

            public_mean_load = 0.0

        # -------------------------------------------------------------
        # Reference electricity price
        # -------------------------------------------------------------

        reference_price = float(
            np.mean(
                _window(
                    public.grid_buy,
                    agent.hour,
                    H
                )
            )
        )

        # -------------------------------------------------------------
        # Cooperative community pressure
        # -------------------------------------------------------------
        #
        # factor = 1.0:
        #     average load
        #
        # factor > 1.0:
        #     above-average community load
        #
        # factor < 1.0:
        #     below-average community load
        #
        # We only penalise high-load periods.
        # -------------------------------------------------------------

        threshold = (
            cfg.coop_peak_threshold_factor
        )

        above_threshold = np.maximum(
            congestion_factor - threshold,
            0.0
        )

        # -------------------------------------------------------------
        # 1. Congestion surcharge
        # -------------------------------------------------------------
        #
        # This is expressed as INR/kWh of imported electricity.
        #
        # Higher public congestion
        # -> higher effective import cost
        # -> stronger incentive to avoid grid import
        # -------------------------------------------------------------

        congestion_surcharge = (
            cfg.coop_congestion_weight
            * cfg.coop_neighbourhood_weight
            * reference_price
            * above_threshold
        )

        # -------------------------------------------------------------
        # 2. Community peak-pressure term
        # -------------------------------------------------------------
        #
        # The existing PlanningProblem supports:
        #
        #       peak_weight[t] * grid_import[t]^2
        #
        # We make that pressure stronger during public high-load
        # periods.
        #
        # This is still computed locally using PUBLIC information.
        # -------------------------------------------------------------

        peak_weight = (
            cfg.coop_peak_weight
            * (
                1.0
                + cfg.coop_neighbourhood_weight
                * above_threshold
            )
        )

        # -------------------------------------------------------------
        # 3. Add public congestion to future import prices
        # -------------------------------------------------------------
        #
        # This makes the community term visible directly to the
        # battery-scheduling search.
        # -------------------------------------------------------------

        cooperative_buy_plan = (
            buy_plan
            + congestion_surcharge
        )

        # -------------------------------------------------------------
        # Build local planning problem
        # -------------------------------------------------------------

        return self._problem(
            agent,
            public,
            cooperative_buy_plan,
            sell_plan,
            congestion=congestion_surcharge,
            peak_weight=peak_weight
        )

    # -----------------------------------------------------------------
    # Cooperative bidding
    # -----------------------------------------------------------------

    def make_bid(
        self,
        agent,
        public: PublicInfo
    ):
        """
        Cooperative agents submit truthful grid-compatible limits.

        Seller:
            ask = grid sell price

        Buyer:
            bid = grid buy price

        This encourages more feasible P2P transactions and avoids
        strategic bid shading dominating the cooperative experiment.
        """

        return self._bid_with_shade(
            agent,
            public,
            shade_sell=0.0,
            shade_buy=0.0
        )


# =====================================================================
# STRATEGY REGISTRY
# =====================================================================

STRATEGIES = {
    "baseline": BaselineStrategy,
    "selfish": SelfishStrategy,
    "cooperative": CooperativeStrategy,
}


# =====================================================================
# FACTORY
# =====================================================================

def make_strategy(
    name: str,
    cfg: Config,
    seed: int
) -> Strategy:

    if name not in STRATEGIES:

        raise ValueError(
            f"unknown strategy '{name}', "
            f"choose from {list(STRATEGIES)}"
        )

    return STRATEGIES[name](
        cfg,
        seed
    )
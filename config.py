"""
config.py - Central, validated configuration for the whole simulation.

Every major simulation parameter lives here so experiments are easy to
re-configure. Money is expressed in Indian rupees (INR), energy in kWh and
power in kW. Since the simulation time step is exactly 1 hour, kW and kWh/h
are used interchangeably.

FOAI concepts represented:
- Dynamic environment
- Stochastic environment
- Agent heterogeneity
- Informed search configuration
- Multi-agent market configuration
- Cooperative vs selfish behavior
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# =====================================================================
# GLOBAL CONSTANTS
# =====================================================================

HOURS_PER_DAY = 24

SCENARIOS = (
    "baseline",
    "selfish",
    "cooperative",
)


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class Config:

    # -----------------------------------------------------------------
    # Simulation scale
    # -----------------------------------------------------------------

    n_households: int = 15
    n_days: int = 7
    horizon: int = 24
    seed: int = 42

    # -----------------------------------------------------------------
    # Time-of-use grid tariff
    # -----------------------------------------------------------------

    offpeak_hours: tuple = (
        23, 0, 1, 2, 3, 4, 5
    )

    peak_hours: tuple = (
        18, 19, 20, 21, 22
    )

    price_offpeak: float = 4.0
    price_shoulder: float = 6.5
    price_peak: float = 10.0

    # Grid pays less for exported electricity than households pay
    # when importing electricity.
    sell_price_ratio: float = 0.45

    # -----------------------------------------------------------------
    # Household heterogeneity
    # -----------------------------------------------------------------

    battery_capacity_range: tuple = (
        4.0,
        14.0
    )

    battery_c_rate: float = 0.3

    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    min_soc_fraction: float = 0.10
    initial_soc_fraction: float = 0.50

    solar_kwp_range: tuple = (
        1.0,
        4.0
    )

    demand_scale_range: tuple = (
        0.5,
        1.5
    )

    # -----------------------------------------------------------------
    # Stochastic environment
    # -----------------------------------------------------------------

    solar_noise_std: float = 0.08

    # Shared daily cloud factor
    cloud_min: float = 0.55

    demand_noise_std: float = 0.15

    # -----------------------------------------------------------------
    # Simulated Annealing
    # -----------------------------------------------------------------

    sa_alpha: float = 0.85

    sa_iters_per_temp: int = 25

    sa_max_iter: int = 600

    sa_patience: int = 300

    sa_t_min_ratio: float = 0.01

    sa_initial_accept_prob: float = 0.8

    sa_calibration_samples: int = 20

    # -----------------------------------------------------------------
    # Battery objective
    # -----------------------------------------------------------------

    degradation_cost: float = 0.05

    terminal_value_factor: float = 0.5

    violation_penalty: float = 1000.0

    # -----------------------------------------------------------------
    # Public P2P price signal
    # -----------------------------------------------------------------

    price_blend_kappa: float = 0.5

    # -----------------------------------------------------------------
    # Cooperative strategy
    # -----------------------------------------------------------------
    #
    # Cooperative agents continue to optimise their own schedules.
    # They do NOT receive other households' private battery states,
    # private valuations, future demand or internal search state.
    #
    # Only public neighbourhood signals are used:
    #
    #   1. neighbourhood congestion
    #   2. aggregate load information
    #   3. public P2P market information
    #
    # -----------------------------------------------------------------

    # Public congestion import-price weight.
    #
    # Higher value means importing during community congestion
    # becomes more expensive in the cooperative objective.
    coop_congestion_weight: float = 0.35

    # Penalty for grid import during congestion.
    coop_peak_weight: float = 0.12

    # Additional community-level pressure.
    coop_neighbourhood_weight: float = 0.30

    # Congestion threshold.
    #
    # 1.0 = average neighbourhood load
    # >1.0 = above average
    # <1.0 = below average
    coop_peak_threshold_factor: float = 1.0

    # -----------------------------------------------------------------
    # Project-level system/social cost
    # -----------------------------------------------------------------
    #
    # This charge is used ONLY during evaluation.
    #
    # System Cost =
    #
    #     Energy Cost
    #     +
    #     Peak Demand Charge
    #
    # It represents the neighbourhood-level impact of peak demand.
    #
    # It is deliberately NOT given directly to a central optimiser.
    # Household agents remain decentralized.
    # -----------------------------------------------------------------

    peak_demand_charge_per_kw: float = 100.0

    # -----------------------------------------------------------------
    # Double Auction
    # -----------------------------------------------------------------

    # Private bid shading range for selfish agents.
    selfish_shade_range: tuple = (
        0.15,
        0.35
    )

    # Clearing-price interpolation:
    #
    # price = ask + auction_k * (bid - ask)
    #
    # 0.5 = midpoint
    auction_k: float = 0.5

    # Minimum trade quantity.
    auction_min_quantity_kwh: float = 0.01

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    results_dir: str = "results"
    plots_dir: str = "plots"
    data_dir: str = "data"
    sample_csv_path: str = "data/sample_profiles.csv"

    # =================================================================
    # VALIDATION
    # =================================================================

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:

        # -------------------------------------------------------------
        # Simulation
        # -------------------------------------------------------------

        if not 1 <= self.n_households <= 200:
            raise ValueError(
                "n_households must be in [1, 200] "
                "(project default is 10-20)"
            )

        if self.n_days < 1:
            raise ValueError(
                "n_days must be >= 1"
            )

        if not 1 <= self.horizon <= 48:
            raise ValueError(
                "horizon must be in [1, 48]"
            )

        # -------------------------------------------------------------
        # Prices
        # -------------------------------------------------------------

        if not (
            self.price_offpeak > 0
            and self.price_shoulder > 0
            and self.price_peak > 0
        ):
            raise ValueError(
                "All electricity prices must be positive"
            )

        if not 0 < self.sell_price_ratio < 1:
            raise ValueError(
                "sell_price_ratio must be in (0,1)"
            )

        # -------------------------------------------------------------
        # Battery
        # -------------------------------------------------------------

        if not (
            0 < self.charge_efficiency <= 1
        ):
            raise ValueError(
                "charge_efficiency must be in (0,1]"
            )

        if not (
            0 < self.discharge_efficiency <= 1
        ):
            raise ValueError(
                "discharge_efficiency must be in (0,1]"
            )

        if not (
            0 <= self.min_soc_fraction < 1
        ):
            raise ValueError(
                "min_soc_fraction must be in [0,1)"
            )

        if not (
            self.min_soc_fraction
            <= self.initial_soc_fraction
            <= 1
        ):
            raise ValueError(
                "initial_soc_fraction must be between "
                "min_soc_fraction and 1"
            )

        # -------------------------------------------------------------
        # Stochastic environment
        # -------------------------------------------------------------

        if self.solar_noise_std < 0:
            raise ValueError(
                "solar_noise_std must be >= 0"
            )

        if self.demand_noise_std < 0:
            raise ValueError(
                "demand_noise_std must be >= 0"
            )

        if not 0 < self.cloud_min <= 1:
            raise ValueError(
                "cloud_min must be in (0,1]"
            )

        # -------------------------------------------------------------
        # Simulated Annealing
        # -------------------------------------------------------------

        if not 0 < self.sa_alpha < 1:
            raise ValueError(
                "sa_alpha must be in (0,1)"
            )

        if self.sa_max_iter < 1:
            raise ValueError(
                "sa_max_iter must be >= 1"
            )

        if self.sa_iters_per_temp < 1:
            raise ValueError(
                "sa_iters_per_temp must be >= 1"
            )

        if self.sa_patience < 1:
            raise ValueError(
                "sa_patience must be >= 1"
            )

        if not 0 < self.sa_t_min_ratio < 1:
            raise ValueError(
                "sa_t_min_ratio must be in (0,1)"
            )

        if not 0 < self.sa_initial_accept_prob < 1:
            raise ValueError(
                "sa_initial_accept_prob must be in (0,1)"
            )

        if self.sa_calibration_samples < 1:
            raise ValueError(
                "sa_calibration_samples must be >= 1"
            )

        # -------------------------------------------------------------
        # Cooperative strategy
        # -------------------------------------------------------------

        if self.coop_congestion_weight < 0:
            raise ValueError(
                "coop_congestion_weight must be >= 0"
            )

        if self.coop_peak_weight < 0:
            raise ValueError(
                "coop_peak_weight must be >= 0"
            )

        if self.coop_neighbourhood_weight < 0:
            raise ValueError(
                "coop_neighbourhood_weight must be >= 0"
            )

        if self.coop_peak_threshold_factor <= 0:
            raise ValueError(
                "coop_peak_threshold_factor must be > 0"
            )

        # -------------------------------------------------------------
        # System cost
        # -------------------------------------------------------------

        if self.peak_demand_charge_per_kw < 0:
            raise ValueError(
                "peak_demand_charge_per_kw must be >= 0"
            )

        # -------------------------------------------------------------
        # Auction
        # -------------------------------------------------------------

        if not 0 <= self.auction_k <= 1:
            raise ValueError(
                "auction_k must be in [0,1]"
            )

        if self.auction_min_quantity_kwh <= 0:
            raise ValueError(
                "auction_min_quantity_kwh must be > 0"
            )

        # -------------------------------------------------------------
        # Household ranges
        # -------------------------------------------------------------

        if (
            self.battery_capacity_range[0]
            <= 0
            or self.battery_capacity_range[1]
            < self.battery_capacity_range[0]
        ):
            raise ValueError(
                "Invalid battery_capacity_range"
            )

        if (
            self.solar_kwp_range[0]
            <= 0
            or self.solar_kwp_range[1]
            < self.solar_kwp_range[0]
        ):
            raise ValueError(
                "Invalid solar_kwp_range"
            )

        if (
            self.demand_scale_range[0]
            <= 0
            or self.demand_scale_range[1]
            < self.demand_scale_range[0]
        ):
            raise ValueError(
                "Invalid demand_scale_range"
            )

    # =================================================================
    # GRID PRICE HELPERS
    # =================================================================

    def grid_buy_prices(self) -> np.ndarray:
        """
        Return the 24-hour electricity BUY price schedule.
        """

        prices = np.full(
            HOURS_PER_DAY,
            self.price_shoulder,
            dtype=float
        )

        prices[
            list(self.offpeak_hours)
        ] = self.price_offpeak

        prices[
            list(self.peak_hours)
        ] = self.price_peak

        return prices

    def grid_sell_prices(self) -> np.ndarray:
        """
        Return the 24-hour electricity SELL price schedule.

        Grid sell price is always lower than grid buy price.
        """

        return (
            self.grid_buy_prices()
            * self.sell_price_ratio
        )

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> dict:
        """
        Convert configuration to a dictionary.

        Used when saving experiment metadata and creating new configs
        for robustness experiments.
        """

        return asdict(self)
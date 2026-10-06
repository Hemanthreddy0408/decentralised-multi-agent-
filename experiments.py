"""
experiments.py - Runs the three experimental configurations, builds
comparison tables, saves CSV/JSON outputs and supports robustness tests.

Configurations:

    baseline
        No local P2P market.

    selfish
        Agents minimise their own electricity cost.

    cooperative
        Agents use public neighbourhood information to reduce community
        congestion and peak load.

The project-level Price of Anarchy is calculated using:

    PoA =
        Selfish System Cost
        -------------------
        Cooperative System Cost

where:

    System Cost =
        Energy Cost
        + Peak Demand Charge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import metrics
from config import Config, SCENARIOS
from data_generator import (
    generate_profiles,
    HouseholdProfile
)
from environment import SmartGridEnvironment


# =====================================================================
# RUN EXPERIMENTS
# =====================================================================

def run_experiments(
    cfg: Config,
    profiles: Optional[list] = None,
    scenarios=SCENARIOS,
    verbose: bool = False
) -> dict:
    """
    Run every scenario using IDENTICAL household profiles.

    Using identical profiles ensures that differences between scenarios
    are caused by strategy/behaviour rather than different input data.
    """

    profiles = (
        profiles
        if profiles is not None
        else generate_profiles(cfg)
    )

    results = {}

    for scenario in scenarios:

        if verbose:
            print(
                f"Running scenario: {scenario}"
            )

        results[scenario] = (
            SmartGridEnvironment(
                cfg,
                profiles,
                scenario
            ).run(
                verbose=verbose
            )
        )

    return results


# =====================================================================
# SUMMARIZE
# =====================================================================

def summarize(
    results: dict
) -> dict:
    """
    Calculate all metrics for all scenarios.

    Also computes:

        - savings versus baseline
        - project Price of Anarchy
        - SOC-adjusted Price of Anarchy
    """

    # -------------------------------------------------------------
    # All scenario metrics
    # -------------------------------------------------------------

    summary = {
        scenario:
            metrics.compute_all(result)
        for scenario, result in results.items()
    }

    # -------------------------------------------------------------
    # Savings relative to baseline
    # -------------------------------------------------------------

    if "baseline" in results:

        baseline = results["baseline"]

        for scenario, result in results.items():

            savings = (
                metrics.per_agent_savings_vs_baseline(
                    result,
                    baseline
                )
            )

            summary[scenario][
                "per_agent_savings_vs_baseline"
            ] = savings.tolist()

            summary[scenario][
                "total_savings_vs_baseline"
            ] = float(
                savings.sum()
            )

    # -------------------------------------------------------------
    # Price of Anarchy
    # -------------------------------------------------------------

    if (
        "selfish" in summary
        and "cooperative" in summary
    ):

        selfish = summary["selfish"]
        cooperative = summary["cooperative"]

        # ---------------------------------------------------------
        # MAIN PROJECT PoA
        # ---------------------------------------------------------
        #
        # Uses System Cost rather than raw electricity bill.
        #
        # This captures the community externality caused by high
        # peak grid usage.
        # ---------------------------------------------------------

        summary["price_of_anarchy"] = (
            metrics.price_of_anarchy(
                selfish["system_cost"],
                cooperative["system_cost"]
            )
        )

        # ---------------------------------------------------------
        # SOC-adjusted PoA
        #
        # Keep this as a diagnostic only.
        # ---------------------------------------------------------

        summary[
            "price_of_anarchy_soc_adjusted"
        ] = metrics.price_of_anarchy(
            selfish["soc_adjusted_cost"],
            cooperative["soc_adjusted_cost"]
        )

        # ---------------------------------------------------------
        # Additional diagnostic:
        # raw electricity-cost PoA
        #
        # This allows us to explain why selfish agents may have a
        # lower cash bill but still have worse neighbourhood welfare.
        # ---------------------------------------------------------

        summary[
            "price_of_anarchy_energy_cost"
        ] = metrics.price_of_anarchy(
            selfish["total_cost"],
            cooperative["total_cost"]
        )

    return summary


# =====================================================================
# COMPARISON TABLE
# =====================================================================

def comparison_table(
    summary: dict
) -> pd.DataFrame:
    """
    Build the final comparison table.

    Main columns:

        Energy Cost
        Peak Demand Charge
        System Cost
        Peak Load
        Grid Dependency
        Renewable Utilization
        Local Trading
        Grid Bought
        Grid Sold
        Fairness
        PoA
    """

    project_poa = summary.get(
        "price_of_anarchy"
    )

    rows = []

    for scenario in SCENARIOS:

        if scenario not in summary:
            continue

        metrics_for_scenario = (
            summary[scenario]
        )

        rows.append({

            "Configuration":
                scenario.capitalize(),

            # -----------------------------------------------------
            # Financial metrics
            # -----------------------------------------------------

            "Energy Cost (INR)":
                round(
                    metrics_for_scenario[
                        "total_cost"
                    ],
                    2
                ),

            "Peak Demand Charge (INR)":
                round(
                    metrics_for_scenario[
                        "peak_demand_charge"
                    ],
                    2
                ),

            "System Cost (INR)":
                round(
                    metrics_for_scenario[
                        "system_cost"
                    ],
                    2
                ),

            # -----------------------------------------------------
            # Grid / energy
            # -----------------------------------------------------

            "Peak Load (kW)":
                round(
                    metrics_for_scenario[
                        "peak_grid_load"
                    ],
                    2
                ),

            "Grid Dependency (%)":
                round(
                    100
                    * metrics_for_scenario[
                        "grid_dependency"
                    ],
                    2
                ),

            "Renewable Util. (%)":
                round(
                    100
                    * metrics_for_scenario[
                        "renewable_utilization"
                    ],
                    2
                ),

            "Local Trading (kWh)":
                round(
                    metrics_for_scenario[
                        "local_energy_traded"
                    ],
                    2
                ),

            "Grid Bought (kWh)":
                round(
                    metrics_for_scenario[
                        "grid_energy_purchased"
                    ],
                    2
                ),

            "Grid Sold (kWh)":
                round(
                    metrics_for_scenario[
                        "grid_energy_sold"
                    ],
                    2
                ),

            # -----------------------------------------------------
            # Fairness
            # -----------------------------------------------------

            "Fairness (Jain)":
                round(
                    metrics_for_scenario[
                        "fairness_jain"
                    ],
                    4
                ),

            # -----------------------------------------------------
            # PoA
            # -----------------------------------------------------

            "PoA (Selfish/Coop)":
                (
                    round(
                        project_poa,
                        4
                    )
                    if (
                        scenario == "selfish"
                        and project_poa is not None
                        and not np.isnan(project_poa)
                    )
                    else "-"
                ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# SAVE OUTPUTS
# =====================================================================

def save_outputs(
    results: dict,
    summary: dict,
    table: pd.DataFrame,
    out_dir: str | Path
) -> None:
    """
    Save:

        comparison table
        summary JSON
        hourly CSVs
        per-agent CSV
        Simulated Annealing statistics
    """

    out = Path(out_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------

    table.to_csv(
        out / "comparison_table.csv",
        index=False
    )

    # -------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------

    with open(
        out / "summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=float
        )

    # -------------------------------------------------------------
    # Hourly + per-agent data
    # -------------------------------------------------------------

    per_agent = None

    for scenario, result in results.items():

        # Hourly data
        result.hourly_dataframe().to_csv(
            out / f"hourly_{scenario}.csv",
            index=False
        )

        # Per-agent data
        df = (
            result
            .agent_dataframe()
            .rename(
                columns=lambda column:
                    column
                    if column in (
                        "agent_id",
                        "tags"
                    )
                    else f"{scenario}_{column}"
            )
        )

        if per_agent is None:

            per_agent = df

        else:

            per_agent = per_agent.merge(
                df.drop(
                    columns=["tags"]
                ),
                on="agent_id"
            )

    # -------------------------------------------------------------
    # Baseline-relative savings
    # -------------------------------------------------------------

    if (
        "baseline" in results
        and per_agent is not None
    ):

        baseline = results["baseline"]

        for scenario, result in results.items():

            per_agent[
                f"{scenario}_savings_vs_baseline_inr"
            ] = (
                metrics.per_agent_savings_vs_baseline(
                    result,
                    baseline
                )
            )

    if per_agent is not None:

        per_agent.to_csv(
            out / "per_agent_results.csv",
            index=False
        )

    # -------------------------------------------------------------
    # Simulated Annealing statistics
    # -------------------------------------------------------------

    with open(
        out / "sa_statistics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                scenario: {
                    key: float(value)
                    for key, value
                    in result.sa_stats.items()
                }
                for scenario, result
                in results.items()
            },
            file,
            indent=2
        )


# =====================================================================
# ROBUSTNESS STUDY
# =====================================================================

def robustness_study(
    base_cfg: Config,
    seeds,
    scenarios=SCENARIOS
) -> pd.DataFrame:
    """
    Repeat the experiment over multiple random seeds.

    Each seed generates a new stochastic neighbourhood but uses the
    SAME profiles across the three strategies for that seed.

    Uses the same metrics as the main experiment, including system cost.
    """

    rows = []

    for seed in seeds:

        cfg = Config(
            **{
                **base_cfg.to_dict(),
                "seed": int(seed)
            }
        )

        results = run_experiments(
            cfg,
            scenarios=scenarios
        )

        summary = summarize(
            results
        )

        for scenario in scenarios:

            if scenario not in summary:
                continue

            current = summary[scenario]

            rows.append({

                "seed":
                    int(seed),

                "scenario":
                    scenario,

                # -------------------------------------------------
                # Cost
                # -------------------------------------------------

                "energy_cost":
                    current["total_cost"],

                "peak_demand_charge":
                    current["peak_demand_charge"],

                "system_cost":
                    current["system_cost"],

                # -------------------------------------------------
                # Grid / energy
                # -------------------------------------------------

                "peak":
                    current["peak_grid_load"],

                "grid_dep":
                    current["grid_dependency"],

                "renewable_util":
                    current["renewable_utilization"],

                "p2p":
                    current["local_energy_traded"],

                # -------------------------------------------------
                # Fairness
                # -------------------------------------------------

                "fairness":
                    current["fairness_jain"],

                # -------------------------------------------------
                # PoA
                # -------------------------------------------------

                "poa":
                    summary.get(
                        "price_of_anarchy",
                        np.nan
                    ),
            })

    return pd.DataFrame(rows)
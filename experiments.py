"""experiments.py - Runs the 3 configurations, builds the comparison table, saves CSV/JSON (PHASE 14)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import metrics
from config import Config, SCENARIOS
from data_generator import generate_profiles, HouseholdProfile
from environment import SmartGridEnvironment


def run_experiments(cfg: Config, profiles: Optional[list] = None, scenarios=SCENARIOS,
                    verbose: bool = False) -> dict:
    """Run every scenario on IDENTICAL data and seeds so that differences are due to behaviour only."""
    profiles = profiles if profiles is not None else generate_profiles(cfg)
    results = {}
    for sc in scenarios:
        if verbose:
            print(f"Running scenario: {sc}")
        results[sc] = SmartGridEnvironment(cfg, profiles, sc).run(verbose=verbose)
    return results


def summarize(results: dict) -> dict:
    """All metrics for all scenarios + Price of Anarchy (+ per-agent savings vs baseline)."""
    summary = {sc: metrics.compute_all(r) for sc, r in results.items()}
    if "baseline" in results:
        for sc, r in results.items():
            sav = metrics.per_agent_savings_vs_baseline(r, results["baseline"])
            summary[sc]["per_agent_savings_vs_baseline"] = sav.tolist()
            summary[sc]["total_savings_vs_baseline"] = float(sav.sum())
    if "selfish" in summary and "cooperative" in summary:
        s, c = summary["selfish"], summary["cooperative"]
        summary["price_of_anarchy"] = metrics.price_of_anarchy(s["total_cost"], c["total_cost"])
        summary["price_of_anarchy_soc_adjusted"] = metrics.price_of_anarchy(
            s["soc_adjusted_cost"], c["soc_adjusted_cost"])
    return summary


def comparison_table(summary: dict) -> pd.DataFrame:
    """Configuration | Total Cost | Peak Load | Grid Dependency | Local Trading | Fairness | PoA"""
    poa = summary.get("price_of_anarchy")
    rows = []
    for sc in SCENARIOS:
        if sc not in summary:
            continue
        m = summary[sc]
        rows.append({
            "Configuration": sc.capitalize(),
            "Total Cost (INR)": round(m["total_cost"], 2),
            "SOC-adj. Cost (INR)": round(m["soc_adjusted_cost"], 2),
            "Peak Load (kW)": round(m["peak_grid_load"], 2),
            "Grid Dependency (%)": round(100 * m["grid_dependency"], 2),
            "Renewable Util. (%)": round(100 * m["renewable_utilization"], 2),
            "Local Trading (kWh)": round(m["local_energy_traded"], 2),
            "Grid Bought (kWh)": round(m["grid_energy_purchased"], 2),
            "Grid Sold (kWh)": round(m["grid_energy_sold"], 2),
            "Fairness (Jain)": round(m["fairness_jain"], 4),
            "PoA (Selfish/Coop)": (round(poa, 4) if sc == "selfish" and poa is not None else "-"),
        })
    return pd.DataFrame(rows)


def save_outputs(results: dict, summary: dict, table: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison_table.csv", index=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    per_agent = None
    for sc, r in results.items():
        r.hourly_dataframe().to_csv(out / f"hourly_{sc}.csv", index=False)
        df = r.agent_dataframe().rename(columns=lambda c: c if c in ("agent_id", "tags") else f"{sc}_{c}")
        per_agent = df if per_agent is None else per_agent.merge(df.drop(columns=["tags"]), on="agent_id")
    if "baseline" in results:
        for sc, r in results.items():
            per_agent[f"{sc}_savings_vs_baseline_inr"] = metrics.per_agent_savings_vs_baseline(r, results["baseline"])
    per_agent.to_csv(out / "per_agent_results.csv", index=False)
    with open(out / "sa_statistics.json", "w", encoding="utf-8") as f:
        json.dump({sc: {k: float(v) for k, v in r.sa_stats.items()} for sc, r in results.items()}, f, indent=2)


def robustness_study(base_cfg: Config, seeds, scenarios=SCENARIOS) -> pd.DataFrame:
    """Repeat the whole experiment for several random seeds (new data each time)."""
    rows = []
    for seed in seeds:
        cfg = Config(**{**base_cfg.to_dict(), "seed": int(seed)})
        s = summarize(run_experiments(cfg, scenarios=scenarios))
        for sc in scenarios:
            rows.append({"seed": seed, "scenario": sc, "total_cost": s[sc]["total_cost"],
                         "peak": s[sc]["peak_grid_load"], "grid_dep": s[sc]["grid_dependency"],
                         "p2p": s[sc]["local_energy_traded"], "fairness": s[sc]["fairness_jain"],
                         "poa": s.get("price_of_anarchy", np.nan)})
    return pd.DataFrame(rows)

"""
data_generator.py - Synthetic (and CSV-supplied) solar / demand data (PHASE 1).

Each household gets
  * a *base* 24-hour solar and demand curve (the typical day the agent "knows"),
  * an *actual* multi-day series = base * noise (what really happens: stochastic),
  * a battery specification drawn from heterogeneous ranges.

CSV format (one row per household-hour, see README):
    household_id,hour,solar_generation,demand
 * hour in 0..23  -> one typical day; it is repeated for all days with random noise.
 * hour in 0..H-1 (H a multiple of 24) -> a full multi-day series is used as "actual".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config, HOURS_PER_DAY

REQUIRED_COLUMNS = ["household_id", "hour", "solar_generation", "demand"]


@dataclass
class HouseholdProfile:
    household_id: int
    tags: list
    solar_base: np.ndarray          # (24,) kWh typical PV output
    demand_base: np.ndarray         # (24,) kWh typical consumption
    solar_actual: np.ndarray        # (n_days*24,)
    demand_actual: np.ndarray       # (n_days*24,)
    battery_capacity: float
    max_charge_rate: float
    max_discharge_rate: float
    charge_efficiency: float
    discharge_efficiency: float
    initial_soc: float
    min_soc_fraction: float


# --------------------------------------------------------------- base curves
def solar_shape() -> np.ndarray:
    """Normalised clear-sky hourly PV curve (sunrise 6h, sunset 18h)."""
    hours = np.arange(HOURS_PER_DAY) + 0.5
    shape = np.sin(np.pi * (hours - 6.0) / 12.0)
    return np.where((hours > 6.0) & (hours < 18.0), np.clip(shape, 0, None) ** 1.3, 0.0)


def demand_shape() -> np.ndarray:
    """Typical residential load: baseline + morning bump + large evening peak."""
    h = np.arange(HOURS_PER_DAY, dtype=float)
    bump = lambda mu, sd: np.exp(-0.5 * ((h - mu) / sd) ** 2)
    return 0.25 + 0.45 * bump(8, 1.5) + 0.20 * bump(13, 3.0) + 0.95 * bump(20, 2.0)


def _random_battery(rng: np.random.Generator, cfg: Config) -> dict:
    capacity = float(rng.uniform(*cfg.battery_capacity_range))
    rate = cfg.battery_c_rate * capacity
    return dict(
        battery_capacity=capacity,
        max_charge_rate=rate,
        max_discharge_rate=rate,
        charge_efficiency=cfg.charge_efficiency,
        discharge_efficiency=cfg.discharge_efficiency,
        initial_soc=cfg.initial_soc_fraction * capacity,
        min_soc_fraction=cfg.min_soc_fraction,
    )


def _tags(demand_scale: float, kwp: float, capacity: float, cfg: Config) -> list:
    lo_d, hi_d = cfg.demand_scale_range
    lo_s, hi_s = cfg.solar_kwp_range
    lo_b, hi_b = cfg.battery_capacity_range
    tags = []
    if demand_scale < lo_d + 0.25 * (hi_d - lo_d):
        tags.append("low-demand")
    elif demand_scale > hi_d - 0.25 * (hi_d - lo_d):
        tags.append("high-demand")
    if kwp < lo_s + 0.25 * (hi_s - lo_s):
        tags.append("low-solar")
    elif kwp > hi_s - 0.25 * (hi_s - lo_s):
        tags.append("high-solar")
    if capacity < lo_b + 0.25 * (hi_b - lo_b):
        tags.append("small-battery")
    elif capacity > hi_b - 0.25 * (hi_b - lo_b):
        tags.append("large-battery")
    return tags or ["average"]


def _daily_cloud_factors(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Weather is shared by the neighbourhood: one cloud factor per day."""
    return rng.uniform(cfg.cloud_min, 1.0, size=cfg.n_days)


def _noisy_series(solar_base, demand_base, cloud, cfg: Config, rng: np.random.Generator):
    """Expand 24-h base curves into n_days of noisy 'actual' data."""
    n_days = len(cloud)
    solar = np.zeros(n_days * HOURS_PER_DAY)
    demand = np.zeros(n_days * HOURS_PER_DAY)
    for d in range(n_days):
        sl = slice(d * HOURS_PER_DAY, (d + 1) * HOURS_PER_DAY)
        s_noise = np.clip(rng.normal(1.0, cfg.solar_noise_std, HOURS_PER_DAY), 0.5, 1.5)
        d_noise = rng.lognormal(0.0, cfg.demand_noise_std, HOURS_PER_DAY)
        solar[sl] = solar_base * cloud[d] * s_noise
        demand[sl] = demand_base * d_noise
    return solar, demand


# ------------------------------------------------------------------ generate
def generate_profiles(cfg: Config) -> list[HouseholdProfile]:
    """Reproducible heterogeneous neighbourhood (same seed -> identical data)."""
    rng = np.random.default_rng(cfg.seed)
    cloud = _daily_cloud_factors(cfg, rng)
    s_shape, d_shape = solar_shape(), demand_shape()
    profiles = []
    for hid in range(cfg.n_households):
        demand_scale = float(rng.uniform(*cfg.demand_scale_range))
        kwp = float(rng.uniform(*cfg.solar_kwp_range))
        battery = _random_battery(rng, cfg)
        # household-specific phase shift of the demand curve (+-1 h) for diversity
        shift = int(rng.integers(-1, 2))
        solar_base = 0.75 * kwp * s_shape
        demand_base = demand_scale * np.roll(d_shape, shift)
        solar_actual, demand_actual = _noisy_series(solar_base, demand_base, cloud, cfg, rng)
        profiles.append(HouseholdProfile(
            household_id=hid,
            tags=_tags(demand_scale, kwp, battery["battery_capacity"], cfg),
            solar_base=solar_base, demand_base=demand_base,
            solar_actual=solar_actual, demand_actual=demand_actual, **battery))
    return profiles


# ----------------------------------------------------------------------- CSV
def save_profiles_csv(profiles: list[HouseholdProfile], path: str | Path, actual: bool = False) -> None:
    """Write profiles in the documented CSV format (base day, or full actual series)."""
    rows = []
    for p in profiles:
        solar, demand = (p.solar_actual, p.demand_actual) if actual else (p.solar_base, p.demand_base)
        for hour in range(len(solar)):
            rows.append((p.household_id, hour, round(float(solar[hour]), 4), round(float(demand[hour]), 4)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)


def load_profiles_csv(path: str | Path, cfg: Config) -> list[HouseholdProfile]:
    """Load user-supplied data. Battery specs are generated from the seed (heterogeneous)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if (df[["solar_generation", "demand"]] < 0).any().any():
        raise ValueError("solar_generation and demand must be non-negative")
    rng = np.random.default_rng(cfg.seed + 1)
    cloud = _daily_cloud_factors(cfg, rng)
    profiles = []
    for new_id, (_, g) in enumerate(df.groupby("household_id", sort=True)):
        g = g.sort_values("hour")
        n = len(g)
        if n % HOURS_PER_DAY != 0 or list(g["hour"]) != list(range(n)):
            raise ValueError("each household needs hours 0..23 (or 0..24*days-1) without gaps")
        solar, demand = g["solar_generation"].to_numpy(float), g["demand"].to_numpy(float)
        solar_base = solar.reshape(-1, HOURS_PER_DAY).mean(axis=0)
        demand_base = demand.reshape(-1, HOURS_PER_DAY).mean(axis=0)
        if n == HOURS_PER_DAY:      # single typical day -> expand with noise
            solar, demand = _noisy_series(solar_base, demand_base, cloud, cfg, rng)
        battery = _random_battery(rng, cfg)
        profiles.append(HouseholdProfile(
            household_id=new_id, tags=["csv"], solar_base=solar_base, demand_base=demand_base,
            solar_actual=solar, demand_actual=demand, **battery))
    return profiles

"""
simulated_annealing.py - Informed (local) search for the battery schedule (PHASE 5).

SEARCH SPACE
    A candidate solution is a list of ``horizon`` integers; entry t is an index into
    LEVELS = (-1, -0.5, 0, +0.5, +1) meaning "discharge at 100/50 %, hold, charge at
    50/100 % of the maximum rate" in hour t.  (-1 = discharge, 0 = hold, +1 = charge
    are the three action types; the two half levels give finer control.)
    Because a single signed number encodes the action, *simultaneous charge and
    discharge is impossible by construction* (constraint 4).

ALGORITHM (all 8 mandatory components are present)
    1. initial solution : previous plan shifted one hour (warm start) or all-HOLD
    2. neighbour        : change one hour / swap two hours / nudge one hour by one level
    3. objective        : ``evaluate_schedule`` (cost + heavy constraint penalties)
    4. temperature init : calibrated so a median worsening move is accepted with p0
    5. acceptance       : always accept improvements, else accept with exp(-delta/T)
    6. cooling          : geometric  T <- alpha * T  every ``iters_per_temp`` moves
    7. stopping         : T < t_min_ratio*T0, or max_iter, or no improvement for ``patience``
    8. best tracking    : the best schedule ever visited is returned (not the last one)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
HOLD = 2                      # index of level 0.0
N_LEVELS = len(LEVELS)
_EPS = 1e-9


# ===================================================================== problem
@dataclass
class PlanningProblem:
    """Everything one agent needs to score a 24-hour battery schedule (its local model)."""
    net_load: Sequence[float]          # forecast demand - solar per hour (kWh, + = deficit)
    buy_price: Sequence[float]         # planning import price (INR/kWh)
    sell_price: Sequence[float]        # planning export price (INR/kWh)
    soc0: float
    capacity: float
    min_soc: float
    max_charge: float
    max_discharge: float
    charge_eff: float
    discharge_eff: float
    degradation_cost: float = 0.0
    terminal_value: float = 0.0        # INR per kWh of stored energy left at the end
    congestion: Optional[Sequence[float]] = None   # cooperative: extra INR per kWh IMPORTED (>= 0)
    peak_weight: Optional[Sequence[float]] = None  # cooperative: quadratic penalty on imports
    violation_penalty: float = 1000.0

    def __post_init__(self) -> None:
        n = len(self.net_load)
        if n == 0 or len(self.buy_price) != n or len(self.sell_price) != n:
            raise ValueError("net_load, buy_price and sell_price must have the same non-zero length")
        if self.congestion is None:
            self.congestion = [0.0] * n
        if self.peak_weight is None:
            self.peak_weight = [0.0] * n
        if len(self.congestion) != n or len(self.peak_weight) != n:
            raise ValueError("congestion / peak_weight length mismatch")
        if not (0 <= self.min_soc <= self.soc0 + _EPS and self.soc0 <= self.capacity + _EPS):
            raise ValueError("soc0 must lie in [min_soc, capacity]")

    @property
    def horizon(self) -> int:
        return len(self.net_load)


# ================================================================== objective
def _roll(schedule: Sequence[int], p: PlanningProblem, trace: Optional[list] = None):
    """
    Simulate a schedule hour by hour. Returns (cost, violation_kwh, final_soc).
    Requests that would break a battery limit are clipped to the physical limit and the
    clipped amount is accumulated as ``violation`` (which is penalised in the objective).
    Energy balance each hour:  grid_flow = net_load + charge_drawn - discharge_delivered.
    """
    soc = p.soc0
    cost = 0.0
    violation = 0.0
    for t in range(p.horizon):
        a = LEVELS[schedule[t]]
        charge = discharge = 0.0
        if a > 0.0:                                   # CHARGE
            charge = a * p.max_charge
            room = (p.capacity - soc) / p.charge_eff
            if charge > room + _EPS:                  # capacity violation
                violation += charge - room
                charge = room if room > 0.0 else 0.0
            soc += charge * p.charge_eff
        elif a < 0.0:                                 # DISCHARGE
            discharge = -a * p.max_discharge
            available = (soc - p.min_soc) * p.discharge_eff
            if discharge > available + _EPS:          # empty-battery violation
                violation += discharge - available
                discharge = available if available > 0.0 else 0.0
            soc -= discharge / p.discharge_eff
        grid = p.net_load[t] + charge - discharge     # + import, - export
        if grid >= 0.0:
            # import: tariff + community terms (congestion surcharge, quadratic peak penalty)
            cost += grid * (p.buy_price[t] + p.congestion[t]) + p.peak_weight[t] * grid * grid
        else:
            cost += grid * p.sell_price[t]            # negative cost = revenue
        cost += p.degradation_cost * (charge + discharge)
        if trace is not None:
            trace.append({"hour": t, "charge": charge, "discharge": discharge, "grid": grid, "soc": soc})
    cost -= p.terminal_value * (soc - p.soc0)         # leftover stored energy has value
    return cost, violation, soc


def evaluate_schedule(schedule: Sequence[int], p: PlanningProblem) -> float:
    """Objective used by SA: electricity cost (+ strategy terms) + heavy penalty on violations."""
    cost, violation, _ = _roll(schedule, p)
    return cost + p.violation_penalty * violation


def schedule_violation(schedule: Sequence[int], p: PlanningProblem) -> float:
    """Total constraint violation in kWh (0.0 means the schedule is feasible)."""
    return _roll(schedule, p)[1]


def simulate_schedule(schedule: Sequence[int], p: PlanningProblem) -> list:
    """Hour-by-hour trace of a schedule (used by tests and for explanations)."""
    trace: list = []
    _roll(schedule, p, trace)
    return trace


# ==================================================================== neighbour
def neighbor(schedule: Sequence[int], rng: random.Random) -> list:
    """Generate a neighbouring schedule with one of three small random moves."""
    new = list(schedule)
    n = len(new)
    r = rng.random()
    if r < 0.55 or n == 1:                            # (a) re-assign one hour
        i = rng.randrange(n)
        new[i] = (new[i] + rng.randrange(1, N_LEVELS)) % N_LEVELS
    elif r < 0.80:                                    # (b) swap two hours (shifts energy in time)
        i, j = rng.sample(range(n), 2)
        new[i], new[j] = new[j], new[i]
    else:                                             # (c) nudge one hour by one level
        i = rng.randrange(n)
        new[i] = min(N_LEVELS - 1, max(0, new[i] + rng.choice((-1, 1))))
    return new


# ======================================================================= SA core
@dataclass
class SAParams:
    alpha: float = 0.85
    iters_per_temp: int = 25
    max_iter: int = 600
    patience: int = 300
    t_min_ratio: float = 0.01
    initial_accept_prob: float = 0.8
    calibration_samples: int = 20

    @classmethod
    def from_config(cls, cfg) -> "SAParams":
        return cls(cfg.sa_alpha, cfg.sa_iters_per_temp, cfg.sa_max_iter, cfg.sa_patience,
                   cfg.sa_t_min_ratio, cfg.sa_initial_accept_prob, cfg.sa_calibration_samples)


@dataclass
class SAResult:
    best_schedule: list
    best_cost: float
    initial_cost: float
    initial_temperature: float
    final_temperature: float
    iterations: int
    accepted: int
    history: list = field(default_factory=list)   # best cost after each temperature level


def calibrate_temperature(objective: Callable, schedule, cost: float, rng: random.Random,
                          params: SAParams) -> float:
    """
    Temperature initialisation: T0 = -median(positive delta) / ln(p0), so that a typical
    worsening move is accepted with probability p0 at the start.  The median (not the mean)
    is used so that a few huge constraint-violation penalties do not inflate T0.
    """
    deltas = []
    for _ in range(params.calibration_samples):
        d = objective(neighbor(schedule, rng)) - cost
        if d > _EPS:
            deltas.append(d)
    if not deltas:
        return 1.0
    deltas.sort()
    return max(1e-6, -deltas[len(deltas) // 2] / math.log(params.initial_accept_prob))


def simulated_annealing(objective: Callable[[Sequence[int]], float], initial: Sequence[int],
                        rng: random.Random, params: SAParams,
                        neighbor_fn: Callable = neighbor) -> SAResult:
    """Generic minimising Simulated Annealing over integer schedules."""
    current = list(initial)
    current_cost = objective(current)
    best, best_cost = list(current), current_cost
    initial_cost = current_cost

    t0 = calibrate_temperature(objective, current, current_cost, rng, params)
    temp = t0
    t_min = params.t_min_ratio * t0
    iteration = accepted = since_improvement = 0
    history = []

    while temp > t_min and iteration < params.max_iter and since_improvement < params.patience:
        for _ in range(params.iters_per_temp):
            if iteration >= params.max_iter:
                break
            iteration += 1
            candidate = neighbor_fn(current, rng)
            cand_cost = objective(candidate)
            delta = cand_cost - current_cost
            # Metropolis acceptance rule
            if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
                current, current_cost = candidate, cand_cost
                accepted += 1
            if current_cost < best_cost - 1e-12:
                best, best_cost = list(current), current_cost
                since_improvement = 0
            else:
                since_improvement += 1
        history.append(best_cost)
        temp *= params.alpha                           # geometric cooling

    return SAResult(best, best_cost, initial_cost, t0, temp, iteration, accepted, history)


def optimize_schedule(problem: PlanningProblem, rng: random.Random, params: SAParams,
                      warm_start: Optional[Sequence[int]] = None) -> SAResult:
    """
    Run SA on one agent's battery-scheduling problem.
    Initial solution = better of {all-HOLD, previous plan shifted by one hour}.
    """
    n = problem.horizon
    objective = lambda s: evaluate_schedule(s, problem)
    candidates = [[HOLD] * n]
    if warm_start is not None and len(warm_start) >= 1:
        shifted = list(warm_start[1:]) + [HOLD]
        candidates.append((shifted + [HOLD] * n)[:n])
    initial = min(candidates, key=objective)
    if problem.capacity <= _EPS:                       # no battery: nothing to optimise
        return SAResult(initial, objective(initial), objective(initial), 0.0, 0.0, 0, 0, [])
    return simulated_annealing(objective, initial, rng, params)

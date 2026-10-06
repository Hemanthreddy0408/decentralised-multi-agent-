import random
import pytest
from simulated_annealing import (PlanningProblem, SAParams, optimize_schedule, evaluate_schedule,
                                 schedule_violation, simulate_schedule, HOLD, LEVELS, neighbor)


def problem(soc0=5.0, net=None):
    # cheap night, expensive evening, surplus solar at noon
    buy = [4.0] * 8 + [6.5] * 10 + [10.0] * 6
    sell = [b * 0.45 for b in buy]
    net = net if net is not None else [0.5] * 8 + [-1.5] * 4 + [0.5] * 6 + [1.5] * 6
    return PlanningProblem(net_load=net, buy_price=buy, sell_price=sell, soc0=soc0, capacity=10.0,
                           min_soc=1.0, max_charge=3.0, max_discharge=3.0, charge_eff=0.95,
                           discharge_eff=0.95, degradation_cost=0.05, terminal_value=3.0)


def test_returns_valid_feasible_solution():
    p = problem()
    res = optimize_schedule(p, random.Random(0), SAParams(max_iter=800))
    assert len(res.best_schedule) == 24
    assert all(0 <= a < len(LEVELS) for a in res.best_schedule)
    assert schedule_violation(res.best_schedule, p) == pytest.approx(0.0)
    assert res.iterations > 0


def test_objective_improves_over_initial_and_never_worse():
    p = problem()
    res = optimize_schedule(p, random.Random(1), SAParams(max_iter=800))
    assert res.best_cost <= res.initial_cost + 1e-9
    assert res.best_cost < res.initial_cost - 1.0        # clear improvement on this instance
    assert res.best_cost == pytest.approx(evaluate_schedule(res.best_schedule, p))


def test_constraint_violations_are_penalised():
    p = problem(soc0=1.0)                                 # battery empty
    bad = [0] * 24                                        # discharge at full rate forever
    assert schedule_violation(bad, p) > 0
    assert evaluate_schedule(bad, p) > evaluate_schedule([HOLD] * 24, p) + 100
    full = problem(soc0=10.0)
    assert schedule_violation([4] * 24, full) > 0         # charging a full battery


def test_trace_respects_soc_limits():
    p = problem()
    res = optimize_schedule(p, random.Random(2), SAParams(max_iter=500))
    for row in simulate_schedule(res.best_schedule, p):
        assert p.min_soc - 1e-9 <= row["soc"] <= p.capacity + 1e-9
        assert not (row["charge"] > 1e-9 and row["discharge"] > 1e-9)


def test_reproducible_with_same_seed():
    p = problem()
    a = optimize_schedule(p, random.Random(5), SAParams(max_iter=300))
    b = optimize_schedule(p, random.Random(5), SAParams(max_iter=300))
    assert a.best_schedule == b.best_schedule


def test_neighbor_changes_schedule_and_keeps_length():
    rng = random.Random(0)
    s = [HOLD] * 24
    assert any(neighbor(s, rng) != s for _ in range(10))
    assert all(len(neighbor(s, rng)) == 24 for _ in range(10))


def test_no_battery_agent_returns_hold_plan():
    p = problem()
    p.capacity = 0.0; p.soc0 = 0.0; p.min_soc = 0.0; p.max_charge = 0.0; p.max_discharge = 0.0
    res = optimize_schedule(p, random.Random(0), SAParams())
    assert res.best_schedule == [HOLD] * 24

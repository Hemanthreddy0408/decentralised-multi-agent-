import math
import numpy as np
import pytest
import metrics
from results import SimulationResult


def fake_result(grid_import, grid_export=None, demand=None, solar=None, cost=None, buy=5.0):
    gi = np.array(grid_import, float)
    T, N = gi.shape
    z = np.zeros((T, N))
    return SimulationResult(
        scenario="x", config={}, agent_ids=list(range(N)), agent_tags=[["a"]] * N,
        solar=z + 1.0 if solar is None else np.array(solar, float),
        demand=z + 2.0 if demand is None else np.array(demand, float),
        soc=z, charge_in=z, discharge_out=z, grid_import=gi,
        grid_export=z if grid_export is None else np.array(grid_export, float),
        p2p_bought=z + 0.5, p2p_sold=z + 0.5,
        agent_cost=gi * buy if cost is None else np.array(cost, float),
        grid_buy_price=np.full(T, buy), grid_sell_price=np.full(T, 0.4 * buy), p2p_price=np.full(T, np.nan),
        initial_soc=np.zeros(N), final_soc=np.zeros(N), inventory_value_per_kwh=np.zeros(N))


def test_total_cost_and_per_agent():
    r = fake_result([[1, 2], [3, 0]])
    assert metrics.total_cost(r) == pytest.approx(30.0)
    assert metrics.per_agent_cost(r).tolist() == [20.0, 10.0]


def test_peak_grid_load():
    r = fake_result([[1, 2], [3, 0], [1, 1]])
    assert metrics.peak_grid_load(r) == pytest.approx(3.0)


def test_grid_dependency():
    r = fake_result([[1, 1], [1, 1]])                  # imports 4, demand 8
    assert metrics.grid_dependency(r) == pytest.approx(0.5)


def test_renewable_utilization_and_local_trading():
    r = fake_result([[0, 0]], grid_export=[[0.25, 0.25]], solar=[[1.0, 1.0]])
    assert metrics.renewable_utilization(r) == pytest.approx(0.75)
    assert metrics.local_energy_traded(r) == pytest.approx(1.0)


def test_jain_fairness():
    assert metrics.jains_index([1, 1, 1, 1]) == pytest.approx(1.0)
    assert metrics.jains_index([1, 0, 0, 0]) == pytest.approx(0.25)    # 1/n worst case
    assert metrics.jains_index([0, 0, 0]) == 1.0                       # zero edge case
    assert metrics.jains_index([5]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.jains_index([])


def test_price_of_anarchy():
    assert metrics.price_of_anarchy(120.0, 100.0) == pytest.approx(1.2)
    assert math.isnan(metrics.price_of_anarchy(10.0, 0.0))


def test_savings_vs_grid_only_reference():
    r = fake_result([[1.0]], demand=[[2.0]], cost=[[4.0]], buy=5.0)    # reference = 10, cost = 4
    assert metrics.per_agent_savings_vs_grid_only(r)[0] == pytest.approx(6.0)

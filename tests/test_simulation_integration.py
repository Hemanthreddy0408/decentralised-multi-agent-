"""End-to-end checks: determinism, energy balance, privacy of broadcasts, decentralisation."""
import dataclasses
import numpy as np
import pytest
from communication import BroadcastMessage
from config import Config
from data_generator import generate_profiles, save_profiles_csv, load_profiles_csv
from experiments import run_experiments
from household_agent import HouseholdAgent
from environment import SmartGridEnvironment

CFG = Config(n_households=6, n_days=2, seed=7, sa_max_iter=150)


@pytest.fixture(scope="module")
def results():
    return run_experiments(CFG)


def test_all_three_scenarios_run(results):
    assert set(results) == {"baseline", "selfish", "cooperative"}


def test_energy_balance_and_nonnegative_flows(results):
    for r in results.values():
        supply = r.solar + r.grid_import + r.p2p_bought + r.discharge_out
        use = r.demand + r.charge_in + r.grid_export + r.p2p_sold
        assert np.allclose(supply, use, atol=1e-6)
        for arr in (r.grid_import, r.grid_export, r.p2p_bought, r.p2p_sold):
            assert (arr >= -1e-9).all()


def test_p2p_conservation_and_baseline_has_no_local_trade(results):
    for sc, r in results.items():
        assert r.p2p_bought.sum() == pytest.approx(r.p2p_sold.sum())
    assert results["baseline"].p2p_bought.sum() == 0.0
    assert results["selfish"].p2p_bought.sum() > 0


def test_soc_limits_hold(results):
    caps = np.array([p.battery_capacity for p in generate_profiles(CFG)])
    for r in results.values():
        assert (r.soc <= caps + 1e-9).all()
        assert (r.soc >= CFG.min_soc_fraction * caps - 1e-9).all()


def test_reproducible():
    a = run_experiments(CFG, scenarios=("selfish",))["selfish"]
    b = run_experiments(CFG, scenarios=("selfish",))["selfish"]
    assert np.allclose(a.agent_cost, b.agent_cost)


def test_grid_sell_price_below_buy_price():
    assert (CFG.grid_sell_prices() < CFG.grid_buy_prices()).all()


def test_broadcast_message_exposes_no_private_state():
    names = {f.name for f in dataclasses.fields(BroadcastMessage)}
    assert names == {"agent_id", "hour", "kind", "quantity"}       # no SOC, valuation, forecast


def test_agents_hold_no_reference_to_other_agents():
    env = SmartGridEnvironment(CFG, generate_profiles(CFG), "cooperative")
    for a in env.agents:
        for v in vars(a).values():
            assert not isinstance(v, HouseholdAgent)


def test_csv_round_trip(tmp_path):
    profiles = generate_profiles(CFG)
    path = tmp_path / "p.csv"
    save_profiles_csv(profiles, path)
    loaded = load_profiles_csv(path, CFG)
    assert len(loaded) == len(profiles)
    assert np.allclose(loaded[0].solar_base, profiles[0].solar_base, atol=1e-3)

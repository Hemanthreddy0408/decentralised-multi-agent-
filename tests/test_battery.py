import pytest
from battery import Battery


def make(soc=5.0):
    return Battery(capacity=10.0, soc=soc, max_charge_rate=2.0, max_discharge_rate=2.0,
                   charge_efficiency=0.9, discharge_efficiency=0.9, min_soc_fraction=0.1)


def test_charging_respects_rate_and_efficiency():
    b = make()
    drawn = b.charge(5.0)
    assert drawn == pytest.approx(2.0)               # rate limit
    assert b.soc == pytest.approx(5.0 + 2.0 * 0.9)   # efficiency loss


def test_discharging_respects_rate_and_efficiency():
    b = make()
    out = b.discharge(5.0)
    assert out == pytest.approx(2.0)
    assert b.soc == pytest.approx(5.0 - 2.0 / 0.9)


def test_capacity_limit():
    b = make(soc=9.9)
    b.charge(2.0)
    assert b.soc <= 10.0 + 1e-9


def test_min_soc_limit():
    b = make(soc=1.05)
    b.discharge(2.0)
    assert b.soc >= b.min_soc - 1e-9


def test_no_simultaneous_charge_discharge():
    with pytest.raises(ValueError):
        make().step(charge_request=1.0, discharge_request=1.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        make().charge(-1)
    with pytest.raises(ValueError):
        Battery(10, 5, 1, 1, charge_efficiency=1.5)
    with pytest.raises(ValueError):
        Battery(10, 20, 1, 1)


def test_round_trip_loses_energy():
    b = make(soc=5.0)
    drawn = b.charge(2.0)                       # stores 2.0 * 0.9 = 1.8 kWh
    delivered = b.discharge(1.8 * 0.9)          # delivering 1.62 kWh uses exactly the stored 1.8 kWh
    assert b.soc == pytest.approx(5.0)
    assert delivered == pytest.approx(1.62) and delivered < drawn   # round-trip loss = 1 - 0.81

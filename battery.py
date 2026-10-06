"""
battery.py - Physical battery model (PHASE 2).

Conventions (1-hour time step, so kW == kWh per step):
  * ``charge(e)``    : ``e`` kWh is DRAWN from the source (solar/grid); only
                       ``e * charge_efficiency`` is stored.
  * ``discharge(e)`` : ``e`` kWh is DELIVERED to the house; the stored energy
                       falls by ``e / discharge_efficiency``.
  * SOC is stored energy in kWh, always kept in [min_soc, capacity].
"""
from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-9


@dataclass
class Battery:
    capacity: float                 # kWh
    soc: float                      # kWh currently stored
    max_charge_rate: float          # kW  (= kWh drawn per hour)
    max_discharge_rate: float       # kW  (= kWh delivered per hour)
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.capacity < 0 or self.max_charge_rate < 0 or self.max_discharge_rate < 0:
            raise ValueError("capacity and rates must be non-negative")
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError("efficiencies must be in (0, 1]")
        if not 0 <= self.min_soc_fraction < 1:
            raise ValueError("min_soc_fraction must be in [0, 1)")
        if not (self.min_soc - _EPS <= self.soc <= self.capacity + _EPS):
            raise ValueError("initial soc must lie within [min_soc, capacity]")

    # ----------------------------------------------------------- properties
    @property
    def min_soc(self) -> float:
        return self.min_soc_fraction * self.capacity

    @property
    def soc_fraction(self) -> float:
        return self.soc / self.capacity if self.capacity > 0 else 0.0

    # ------------------------------------------------------------- limits
    def max_charge_input(self) -> float:
        """Largest energy (kWh) that may be drawn from the source this hour."""
        headroom = (self.capacity - self.soc) / self.charge_efficiency
        return max(0.0, min(self.max_charge_rate, headroom))

    def max_discharge_output(self) -> float:
        """Largest energy (kWh) that may be delivered to the house this hour."""
        available = (self.soc - self.min_soc) * self.discharge_efficiency
        return max(0.0, min(self.max_discharge_rate, available))

    # ------------------------------------------------------------ actions
    def charge(self, energy: float) -> float:
        """Charge with ``energy`` kWh drawn from the source. Returns energy actually drawn."""
        if energy < 0:
            raise ValueError("charge energy must be >= 0")
        drawn = min(energy, self.max_charge_input())
        self.soc = min(self.capacity, self.soc + drawn * self.charge_efficiency)
        return drawn

    def discharge(self, energy: float) -> float:
        """Request ``energy`` kWh delivered to the house. Returns energy actually delivered."""
        if energy < 0:
            raise ValueError("discharge energy must be >= 0")
        delivered = min(energy, self.max_discharge_output())
        self.soc = max(self.min_soc, self.soc - delivered / self.discharge_efficiency)
        return delivered

    def step(self, charge_request: float = 0.0, discharge_request: float = 0.0) -> tuple[float, float]:
        """
        Apply one hour of battery operation.
        Constraint 4: simultaneous charging and discharging is forbidden.
        Returns (energy_drawn_for_charging, energy_delivered_by_discharging).
        """
        if charge_request > _EPS and discharge_request > _EPS:
            raise ValueError("simultaneous charge and discharge is not allowed")
        if charge_request > _EPS:
            return self.charge(charge_request), 0.0
        if discharge_request > _EPS:
            return 0.0, self.discharge(discharge_request)
        return 0.0, 0.0

from Simulations.PowerSystemSimulation.LiIonBatteryCell import BatteryCell

class MockBatteryCell(BatteryCell):
    """
    A mock battery cell that bypasses the pybamm simulation.
    It provides a simplified model with hardcoded values.
    """
    def __init__(self, capacity=3.5, starting_soc=1.0):
        self._voltage = 3.7  # Nominal voltage
        self._current = 0
        self._capacity = capacity
        self._soc = starting_soc

    def get_voltage(self):
        # Return a slightly varying voltage based on SoC
        return self._voltage * (0.8 + 0.2 * self._soc)

    def get_current(self):
        return self._current

    def get_state_of_charge(self) -> float:
        return self._soc

    def set_current(self, current):
        # A very simple SoC update. This doesn't account for time.
        # For a running simulation, this is a simplification.
        if self._current > 0: # Discharging
            self._soc -= 0.0001
        elif self._current < 0: # Charging
            self._soc += 0.0001
        
        # Clamp SoC between 0 and 1
        self._soc = max(0.0, min(1.0, self._soc))
        
        self._current = current

    def get_nominal_voltage(self):
        return 3.7

    def cutoff_voltage(self):
        return 3.0
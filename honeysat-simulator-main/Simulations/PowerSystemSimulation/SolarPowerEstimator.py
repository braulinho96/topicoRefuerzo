from abc import ABC, abstractmethod

from Simulations.OrbitalSimulation import OrbitalSimulation


class SolarPowerEstimator(ABC):
    @abstractmethod
    def get_power(self) -> float:
        pass


class ConstantPowerEstimator(SolarPowerEstimator):
    __osim: OrbitalSimulation
    __power: float

    # The power argument in the constructor below is how much power from the solar panels can be output.
    def __init__(self, power: float, o_sim: OrbitalSimulation):
        self.__power = power
        self.__osim = o_sim

    def get_power(self) -> float:
        return self.__power if self.__osim.send_request('is_sun_present').result()['is_sunlit'] else 0.0

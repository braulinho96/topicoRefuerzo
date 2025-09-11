from random import random, uniform
from typing import Callable

from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData

from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.PowerSystemSimulation.PowerSystemSimulation import PowerSystemSimulation


class CurrentSensor(Sensor):
    """
    CURRENT SENSOR TO FEED CURRENT DATA TO THE FLIGHT SOFTWARE
    A sensor that returns the present current based on the PowerSystemSimulation.
    """

    def __init__(self, power_system_simulation: PowerSystemSimulation, noise_level=0.1):
        """
        Initialize the sensor.

        :param power_system_simulation: The PowerSystemSimulation instance to use
        """

        super().__init__()

        self.sensor_name = "CurrentSensor"

        if not power_system_simulation:
            raise ValueError("No power_system_simulation available")
        if not isinstance(power_system_simulation, PowerSystemSimulation):
            raise TypeError("Simulation must be a PowerSystemSimulation")

        self.power_system_simulation = power_system_simulation
        self.noise_level = noise_level

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('current', self.measure_current))
        list_of_methods.append(('current_in', self.measure_current_in))
        list_of_methods.append(('current_out', self.measure_current_out))

        return list_of_methods

    def measure_current(self) -> SensorData:
        """
        Get the simulated current measurement from the Power System Simulation.

        Returns:
        - float: Simulated current reading.
        """

        current = self.power_system_simulation.get_current()
        # Simulate a small amount of random noise
        noise = uniform(-self.noise_level, self.noise_level)

        # Add noise to the simulated current
        current = current + noise

        return SensorData("Current", current)

    def measure_current_out(self) -> SensorData:
        """
        Get the simulated current measurement from the Power System Simulation.

        Returns:
        - float: Simulated current reading.
        """

        current_out = self.power_system_simulation.get_power_out() / self.power_system_simulation.get_voltage()

        return SensorData("Current Out", current_out)

    def measure_current_in(self) -> SensorData:
        """
        Get the simulated current measurement from the Power System Simulation.

        Returns:
        - float: Simulated current reading.
        """

        current_in = self.power_system_simulation.get_power_in() / self.power_system_simulation.get_voltage()

        return SensorData("Current In", current_in)

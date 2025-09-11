from random import random, uniform
from typing import Callable
from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.PowerSystemSimulation.PowerSystemSimulation import PowerSystemSimulation


class VoltageSensor(Sensor):
    """
    VOLTAGE SENSOR TO FEED VOLTAGE DATA TO THE FLIGHT SOFTWARE
    A sensor that returns the current voltage based on the PowerSystemSimulation.
    """

    def __init__(self, power_system_simulation: PowerSystemSimulation, noise_level=0.1):
        """
        Initialize the sensor.

        :param power_system_simulation: The PowerSystemSimulation instance to use
        """

        super().__init__()

        self._sensor_name = "VoltageSensor"

        if not power_system_simulation:
            raise ValueError("No power_system_simulation available")
        if not isinstance(power_system_simulation, PowerSystemSimulation):
            raise TypeError("Simulation must be a PowerSystemSimulation")

        self.power_system_simulation = power_system_simulation
        self.noise_level = noise_level

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('voltage', self.measure_voltage))

        return list_of_methods

    def measure_voltage(self) -> SensorData:
        """
        Get the simulated current measurement from the Power System Simulation.

        Returns:
        - float: Simulated current reading.
        """

        # voltage = self.power_system_simulation.send_request('get_voltage').result()['voltage']
        voltage = self.power_system_simulation.get_voltage()
        # Simulate a small amount of random noise
        noise = uniform(-self.noise_level, self.noise_level)

        # Add noise to the simulated current
        voltage = voltage + noise

        # Simulation output is in Volts but SUCHAI wants mV.
        voltage *= 1000

        return SensorData("Voltage", voltage)

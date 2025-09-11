from random import uniform
from typing import Callable

from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Simulations.ThermalSimulation.ThermalSimulation import ThermalSimulation


class TemperatureSensor(Sensor):
    """
    A sensor that estimates the temperature based on the thermal simulation
    """
    __thermal_simulation: ThermalSimulation

    def __init__(self, thermal_simulation: ThermalSimulation):
        """
        Initialize the sensor.

        :param thermal_simulation: The ThermalSimulation instance to use
        """

        super().__init__()

        self._sensor_name = "TemperatureSensor"

        if not thermal_simulation:
            raise ValueError("No thermal_simulation available")
        if not isinstance(thermal_simulation, ThermalSimulation):
            raise TypeError("Simulation must be a ThermalSimulation")

        self.__thermal_simulation = thermal_simulation

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('temperature', self.get_value))

        return list_of_methods

    def get_value(self) -> SensorData:
        """
        After retrieving the temperature from the simulation, adds random noise to the value and returns it.

        :return: The temperature in Kelvin as a float
        """
        base_temperature = self.__thermal_simulation.send_request('get_temperature').result()['temperature']

        MIN_TEMPERATURE = 268  # Minimum temperature in Kelvin
        MAX_TEMPERATURE = 320  # Maximum temperature in Kelvin
        # Adding random noise
        temperature_with_noise = base_temperature + uniform(-5, 5)

        # Ensuring the temperature is within the defined range
        final_temperature = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, temperature_with_noise))

        return SensorData("Temperature", final_temperature)

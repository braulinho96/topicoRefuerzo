from random import random, uniform
from typing import Callable, List, Tuple, Any
import random
from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Simulations.OrbitalSimulation import OrbitalSimulation
from SatellitePersonality import SatellitePersonality


class GStoFSCommSensor(Sensor):
    """
    Communications Sensor that will enable or disable communication between the Flight Software and the Ground Station.
    Even though it is  using the Sensor ABC, it functions more as an actuator.
    """

    def __init__(self, orbital_simulation: OrbitalSimulation):
        """
        Initialize the sensor.

        """

        super().__init__()

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be a OrbitalSimulation")

        self.orbital_simulation = orbital_simulation
        self._sensor_name = "GStoFSCommSensor"

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('is_comm_reachable', self.is_comm_reachable))

        # TODO add missing values

        return list_of_methods

    def _generate_sensor_list_of_settable_methods(self) -> list[tuple[str, Callable[[Any], None]]]:

        list_of_settable_methods = []
        list_of_settable_methods.append(('set_comm_reachable', self.set_ground_station_comm_reachability))

        # TODO add missing values

        return list_of_settable_methods

    def is_comm_reachable(self) -> SensorData:
        reachable_data = self.orbital_simulation.send_request('get_comm_reachable_data').result()
        alt = reachable_data['satellite_altitude_degrees']
        distance = reachable_data['satellite_altitude_distance']
        # Convert distance to float
        distance = float('{:.1f}'.format(distance.km))
        # signal_power = self.inverse_square_law(distance, self.initial_power)
        # Check that satellite is over the horizon

        altitude_angle = alt.degrees

        threshold_1 = SatellitePersonality.THRESHOLD_1 + self.generate_random_noise()
        threshold_2 = SatellitePersonality.THRESHOLD_2 + self.generate_random_noise()
        threshold_3 = SatellitePersonality.THRESHOLD_3 + self.generate_random_noise()
        threshold_4 = SatellitePersonality.THRESHOLD_4 + self.generate_random_noise()

        probability = 0.0
        is_reachable = False

        if altitude_angle <= 0:
            probability = SatellitePersonality.PROBABILITY_1
            is_reachable = False
        elif 0 < altitude_angle <= threshold_1:
            probability = SatellitePersonality.PROBABILITY_2
            is_reachable = self.decision(probability)
        elif threshold_1 <= altitude_angle <= threshold_2:
            probability = SatellitePersonality.PROBABILITY_3
            is_reachable = self.decision(probability)
        elif threshold_2 <= altitude_angle <= threshold_3:
            probability = SatellitePersonality.PROBABILITY_4
            is_reachable = self.decision(probability)
        elif threshold_3 <= altitude_angle <= threshold_4:
            probability = SatellitePersonality.PROBABILITY_5
            is_reachable = self.decision(probability)
        elif threshold_4 <= altitude_angle <= 90:
            probability = SatellitePersonality.PROBABILITY_6
            is_reachable = self.decision(probability)
        # TODO Log this?
        # print(f"{self._sensor_name} -> altitude_angle: {altitude_angle}, distance: {distance}, alt: {alt}, reachable: {is_reachable}")
        return SensorData("is_reachable", is_reachable)

    def decision(self, probability):
        return random.random() < probability

    def generate_random_noise(self, min_value=-1.0, max_value=1.0):
        """
        Generate random noise as a floating-point number within a specified range.

        Parameters:
            min_value (float): Minimum value of the random noise (default is 0.0).
            max_value (float): Maximum value of the random noise (default is 1.0).

        Returns:
            float: Random noise sample.
        """
        return random.uniform(min_value, max_value)

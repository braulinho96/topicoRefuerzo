from typing import Callable

import numpy as np

from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData

from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.RotationSimulation import RotationSimulation

import scipy as sp
from scipy.spatial.transform import Rotation as R


class AccelerometerSensor(Sensor):
    """
    Sensor that provides acceleration-related data in an orbital context.
    """

    def __init__(self, orbital_simulation: OrbitalSimulation):
        """
        Initialize the sensor.

        :param orbital_simulation: The OrbitalSimulation instance to use
        """

        super().__init__()
        self._sensor_name = "AccelerometerSensor"

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be an OrbitalSimulation")

        self.orbital_simulation = orbital_simulation

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('acceleration', self.get_acceleration))

        return list_of_methods

    # TODO should we refactor the get_all_data method to get_acceleration?
    def get_acceleration(self) -> SensorData:
        """
        After calling the Orbital, it returns the acceleration-related data.
        """

        earth_radius = 6371

        orientation_with_respect_to_earth = self.orbital_simulation.send_request("get_earth_orientation").result()[
            'orbital_earth_orientation']
        distance_from_earth = self.orbital_simulation.send_request("get_distance_to_earth_center").result()[
            'distance_to_earth_center']

        current_gravitational_acceleration = sp.constants.g * pow((earth_radius / distance_from_earth), 2)
        acceleration = np.array([0, 0, -current_gravitational_acceleration])

        # https://motion.cs.illinois.edu/RoboticSystems/
        earth_rotation = R.from_euler('xyz', orientation_with_respect_to_earth, False)

        acceleration_wrt_earth = earth_rotation.inv().apply(acceleration)

        return SensorData("Acceleration", acceleration_wrt_earth)

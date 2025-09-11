from typing import Callable

from scipy.spatial.transform import Rotation as R

from SatellitePersonality import SatellitePersonality
from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Sensors.SunSensor import SunSensor
from Simulations.OrbitalSimulation import OrbitalSimulation
import numpy as np


class EveryFaceSunSensor(Sensor):
    """
    A sensor that estimates the presence of the sun and the satellite orientation with respect to the sun.
    """

    def __init__(self, orbital_simulation: OrbitalSimulation):
        """
        Initialize the sensor.

        :param orbital_simulation: The OrbitalSimulation instance to use
        """

        super().__init__()
        self._sensor_name = "EveryFaceSunSensor"

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be a OrbitalSimulation")

        self.orbital_simulation = orbital_simulation

        self._sun_sensor_list = []
        for i in range(6):
            self._sun_sensor_list.append(
                SunSensor(orbital_simulation, SatellitePersonality.MULTIPLE_SUN_SENSOR_ORIENTATION[i]))

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('in_sunlight', self.is_satellite_in_sunlight))
        list_of_methods.append(('orientation_from_sun', self.sun_orientation_wrt_satellite))
        list_of_methods.append(('sun_angle_incidence', self.sun_angle_incidence))
        list_of_methods.append(('sun_angle_incidence_as_current', self.sun_angle_incidence_as_current))

        return list_of_methods

    def is_satellite_in_sunlight(self) -> SensorData:
        """
        After calling the OrbitalSimulation for a first check, then calls all the sun sensors to check if the satellite
        is in sunlight. If at least one sensor is in sunlight, it returns True, False otherwise.

        :return: True or False if the satellite is in sunlight
        """

        is_sun_present = self._check_sun_light_from_simulation()

        # If not present, useless to call all the sensors
        if not is_sun_present:
            return SensorData("Sunlight", is_sun_present)

        # if at least one sensor is in sunlight, return True
        for sensor in self._sun_sensor_list:
            if sensor.is_satellite_in_sunlight().value:
                return SensorData("Sunlight", True)

        return SensorData("Sunlight", False)

    def sun_orientation_wrt_satellite(self, sensor_id: int = None) -> SensorData:
        """
        After calling OrbitalSimulation, it returns the satellite orientation with respect to the sun sensor.
        If the satellite is not in sunlight, it returns None and "Error" as data type.
        If the sensor_id is an integer between 0 and 5, return the orientation of the specified sensor.
        If no sensor_id is provided, return the orientation of all the sensors.
        """

        # https://motion.cs.illinois.edu/RoboticSystems/

        # if the satellite is not in sunlight return None and "Error" as data type
        if not self._check_sun_light_from_simulation():
            return SensorData("Sun orientation", None, "Error")

        # if the sensor_id is an integer between 0 and 5, return the orientation of the specified sensor
        if isinstance(sensor_id, int) and 0 <= sensor_id < 6:
            return self._sun_sensor_list[sensor_id].sun_orientation_wrt_satellite()

        # if no sensor_id is provided, return the orientation of all the sensors
        elif sensor_id is None:
            orientation = []
            for sensor in self._sun_sensor_list:
                orientation.append(sensor.sun_orientation_wrt_satellite().value)
            return SensorData("Sun orientation", orientation, type(orientation))
        else:
            raise ValueError("Invalid sensor_id")

    def sun_angle_incidence(self, sensor_id: int = None) -> SensorData:
        """
        After calling OrbitalSimulation, it returns the angle of incidence of the sun with respect to the sensor.
        If the satellite is not in sunlight, it returns None and "Error" as data type.
        :return: The angle of incidence of the sun with respect to the sensor
        """

        # if the satellite is not in sunlight return None and "Error" as data type
        if not self._check_sun_light_from_simulation():
            return SensorData("Sun angle incidence", None, "Error")

        # if the sensor_id is an integer between 0 and 5, return the orientation of the specified sensor
        if isinstance(sensor_id, int) and 0 <= sensor_id < 6:
            return self._sun_sensor_list[sensor_id].sun_angle_incidence()

        # if no sensor_id is provided, return the orientation of all the sensors
        elif sensor_id is None:
            sun_angle_incidence = []
            for sensor in self._sun_sensor_list:
                sun_angle_incidence.append(sensor.sun_angle_incidence().value)
            return SensorData("Sun orientation", sun_angle_incidence, type(sun_angle_incidence))
        else:
            raise ValueError("Invalid sensor_id")

    def sun_angle_incidence_as_current(self, sensor_id: int = None) -> SensorData:
        """
        After calling OrbitalSimulation, it returns the angle of incidence of the sun with respect to the sensor.
        If the satellite is not in sunlight, it returns None and "Error" as data type.
        :return: The angle of incidence of the sun with respect to the sensor
        """

        # if the satellite is not in sunlight return None and "Error" as data type
        if not self._check_sun_light_from_simulation():
            return SensorData("Sun angle incidence", None, "Error")

        # if the sensor_id is an integer between 0 and 5, return the orientation of the specified sensor
        if isinstance(sensor_id, int) and 0 <= sensor_id < 6:
            return self._sun_sensor_list[sensor_id].sun_angle_incidence_as_current()

        # if no sensor_id is provided, return the orientation of all the sensors
        elif sensor_id is None:
            sun_angle_incidence_as_current = []
            for sensor in self._sun_sensor_list:
                sun_angle_incidence_as_current.append(sensor.sun_angle_incidence_as_current().value)
            return SensorData("Sun orientation", sun_angle_incidence_as_current, type(sun_angle_incidence_as_current))
        else:
            raise ValueError("Invalid sensor_id")

    def _check_sun_light_from_simulation(self) -> bool:
        """
        Check the simulation if the satellite is in sunlight at all.

        :return: True if the satellite is in sunlight, False otherwise
        """
        return self.orbital_simulation.send_request('is_sun_present').result()['is_sunlit']

    # def _sun_in_sensor_visibility_cone(self, np.array):

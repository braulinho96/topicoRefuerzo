from typing import Callable

from scipy.spatial.transform import Rotation as R

from SatellitePersonality import SatellitePersonality
from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Simulations.OrbitalSimulation import OrbitalSimulation
import numpy as np


class SunSensor(Sensor):
    """
    A sensor that estimates the presence of the sun and the satellite orientation with respect to the sun.
    """

    def __init__(self, orbital_simulation: OrbitalSimulation,
                 sun_sensor_orientation: tuple[int, int, int] = SatellitePersonality.SUN_SENSOR_ORIENTATION):
        """
        Initialize the sensor.

        :param orbital_simulation: The OrbitalSimulation instance to use
        """

        super().__init__()
        self._sensor_name = "SunSensor"

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be a OrbitalSimulation")

        self._orbital_simulation = orbital_simulation

        if not sun_sensor_orientation or not isinstance(sun_sensor_orientation, tuple) or len(
                sun_sensor_orientation) != 3:
            raise ValueError("No sun_sensor_orientation available")
        self._sun_sensor_orientation = np.array(sun_sensor_orientation)

        # Store max angle of visibility from Satellite personality field of view
        self._max_angle_of_visibility_radians = (SatellitePersonality.SUN_SENSOR_FIELD_OF_VIEW / 2) * np.pi / 180

        # store min and max current
        self._min_current = SatellitePersonality.SUN_SENSOR_MIN_CURRENT
        self._max_current = SatellitePersonality.SUN_SENSOR_MAX_CURRENT

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('in_sunlight', self.is_satellite_in_sunlight))
        list_of_methods.append(('orientation_from_sun', self.sun_orientation_wrt_satellite))
        list_of_methods.append(('sun_angle_incidence', self.sun_angle_incidence))
        list_of_methods.append(('sun_angle_incidence_as_current', self.sun_angle_incidence_as_current))

        return list_of_methods

    def is_satellite_in_sunlight(self) -> SensorData:
        """
        After retrieving the  presence of the sun from the OrbitalSimulation instance, it returns True if the satellite
        sun sensor is in sunlight, False otherwise.

        :return: True or False if the satellite is in sunlight
        """

        is_sun_present = self._check_sun_light()

        if not is_sun_present:
            return SensorData("Sunlight", False, "bool")

        sun_orientation_wrt_sensor = self._get_sun_orientation_wrt_sensor()

        # check if the sun is in the visibility cone of the sensor
        if not self._sun_in_sensor_visibility_cone(sun_orientation_wrt_sensor):
            return SensorData("Sunlight", False, "bool")

        return SensorData("Sunlight", True, "bool")

    def sun_orientation_wrt_satellite(self) -> SensorData:
        """
        After calling OrbitalSimulation, it returns the satellite orientation with respect to the sun sensor.
        If the satellite is not in sunlight, it returns None and "Error" as data type.
        :return: The satellite orientation with respect to the sun
        """

        # https://motion.cs.illinois.edu/RoboticSystems/

        # if the satellite is not in sunlight return None and "Error" as data type
        if not self._check_sun_light():
            return SensorData("Sun orientation", None, "Error")

        sun_orientation_wrt_sensor = self._get_sun_orientation_wrt_sensor()

        # check if the sun is in the visibility cone of the sensor
        if not self._sun_in_sensor_visibility_cone(sun_orientation_wrt_sensor):
            return SensorData("Sun orientation", None, "Error")

        return SensorData("Sun orientation", sun_orientation_wrt_sensor,
                          type(sun_orientation_wrt_sensor))

    def sun_angle_incidence(self) -> SensorData:
        """
        After calling OrbitalSimulation, it returns the angle of incidence of the sun with respect to the sensor.
        If the satellite is not in sunlight, it returns None and "Error" as data type.
        :return: The angle of incidence of the sun with respect to the sensor
        """

        # if the satellite is not in sunlight return None and "Error" as data type
        if not self._check_sun_light():
            return SensorData("Sun angle incidence", None, "Error")

        sun_orientation_wrt_sensor = self._get_sun_orientation_wrt_sensor()

        # check if the sun is in the visibility cone of the sensor
        if not self._sun_in_sensor_visibility_cone(sun_orientation_wrt_sensor):
            return SensorData("Sun angle incidence", None, "Error")

        angle_radians = self._angle_of_incidence(sun_orientation_wrt_sensor)
        return SensorData("Sun angle incidence", angle_radians, "float")

    def sun_angle_incidence_as_current(self) -> SensorData:
        """
        After calling OrbitalSimulation, it returns the angle of incidence of the sun with respect to the sensor.
        If the satellite is not in sunlight, it returns None and "Error" as data type.
        :return: The angle of incidence of the sun with respect to the sensor
        """

        # if the satellite is not in sunlight return None and "Error" as data type
        if not self._check_sun_light():
            return SensorData("Sun angle incidence", None, "Error")

        sun_orientation_wrt_sensor = self._get_sun_orientation_wrt_sensor()

        # check if the sun is in the visibility cone of the sensor
        if not self._sun_in_sensor_visibility_cone(sun_orientation_wrt_sensor):
            return SensorData("Sun angle incidence", None, "Error")

        angle_radians = self._angle_of_incidence(sun_orientation_wrt_sensor)

        # Convert angle to current
        current = self._angle_to_current(angle_radians)

        return SensorData("Sun angle incidence", int(current), "int")

    def _check_sun_light(self) -> bool:
        """
        Check if the satellite is in sunlight at all.

        :return: True if the satellite is in sunlight, False otherwise
        """

        return self._orbital_simulation.send_request('is_sun_present').result()['is_sunlit']

    def _get_sun_orientation_wrt_sensor(self) -> np.array:
        satellite_orientation = self._orbital_simulation.send_request('get_sun_orientation').result()[
            'orbital_sun_orientation']
        sun_sensor_orientation = R.from_euler('xyz', self._sun_sensor_orientation, False)

        sun_orientation_wrt_sensor = sun_sensor_orientation.inv().apply(satellite_orientation)
        return sun_orientation_wrt_sensor

    def _angle_of_incidence(self, sun_orientation_wrt_sensor: np.array) -> float:
        normal_vector = np.array([0, 0, 1])

        if sun_orientation_wrt_sensor is None or not isinstance(sun_orientation_wrt_sensor, np.ndarray):
            raise ValueError("No sun_orientation_wrt_sensor available")

        normalized_sun_orientation_wrt_sensor = sun_orientation_wrt_sensor / np.linalg.norm(
            sun_orientation_wrt_sensor)
        normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

        dot_product = np.dot(normalized_sun_orientation_wrt_sensor, normalized_normal_vector)
        angle_radians = np.arccos(dot_product)
        return angle_radians

    def _sun_in_sensor_visibility_cone(self, sun_orientation_wrt_sensor: np.array) -> bool:
        angle_radians = self._angle_of_incidence(sun_orientation_wrt_sensor)
        return angle_radians < self._max_angle_of_visibility_radians

    def _angle_to_current(self, angle_incidence_radians: float) -> float:
        """
        Max current corresponds to 0 degrees
        Min current corresponds to max field of view
        """

        if angle_incidence_radians < 0 or angle_incidence_radians > self._max_angle_of_visibility_radians:
            raise ValueError("Angle is out of the valid range")

        # Convert angle to a ratio between 0 and 1
        ratio = angle_incidence_radians / self._max_angle_of_visibility_radians

        # Linearly interpolate between the max and min current values
        current = (1 - ratio) * self._max_current + ratio * self._min_current
        return current

from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from SatellitePersonality import SatellitePersonality
from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Simulations.MagneticSimulation import MagneticSimulation
from Simulations.RotationSimulation import RotationSimulation


class MagnetometerSensor(Sensor):
    """
    A sensor that estimates the presence and intensity of the magnetic field
    """

    def __init__(self, magnetic_simulation: MagneticSimulation, rotation_simulation: RotationSimulation):
        """
        Initialize the sensor.

        :param orbital_simulation: The OrbitalSimulation instance to use
        """

        super().__init__()
        self._sensor_name = "MagnetometerSensor"

        if not magnetic_simulation or not isinstance(magnetic_simulation, MagneticSimulation):
            raise ValueError("No magnetic_simulation available")
        self.magnetic_simulation = magnetic_simulation

        if not rotation_simulation or not isinstance(rotation_simulation, RotationSimulation):
            raise ValueError("No rotation_simulation available")
        self.rotation_simulation = rotation_simulation

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('magnetic_field', self.get_earth_magnetic_field))
        list_of_methods.append(('magnetic_field_mGauss', self.get_magnetic_field_mGauss))

        return list_of_methods

    def get_earth_magnetic_field(self) -> SensorData:
        """
        After retrieving the magnetic field from the MagneticSimulation instance, it returns the magnetic field adjusted
        to the satellite orientation.
        :return: The magnetic field adjusted to the satellite orientation in nT
        """

        magnetic_field_response = self.magnetic_simulation.send_request('earth_magnetic_field').result()

        magnetic_field = np.array(
            [magnetic_field_response['north'], magnetic_field_response['east'], magnetic_field_response['vertical']])

        satellite_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        # Euler angular orientation 3-2-1 (yaw-pitch-roll) with respec to to inertial frame
        satellite_orientation = np.array(self.rotation_simulation.send_request('quaternion').result())
        satellite_orientation[:3] *= -1  # conjugate
        rotation_satellite = R.from_quat(satellite_orientation)
        # rotation_satellite = R.from_euler('xyz', satellite_orientation, False)
        body_frame_earth_magnetic_field = rotation_satellite.apply(magnetic_field)
        return SensorData("Magnetic field", tuple(body_frame_earth_magnetic_field))

    def get_magnetic_field_mGauss(self) -> SensorData:
        """
        Internally calls the get_earth_magnetic_field method and converts the magnetic field from nT to mGauss.
        :return: The magnetic field adjusted to the satellite orientation in mGauss
        """

        magnetic_field = np.array(self.get_earth_magnetic_field().value)
        magnetic_field_mGauss = magnetic_field * 0.01

        return SensorData("Magnetic field mGauss", tuple(magnetic_field_mGauss))

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

        # TODO ADD CHECK FOR VISIBILITY CONE
        satellite_orientation = self.orbital_simulation.send_request('get_sun_orientation').result()[
            'orbital_sun_orientation']
        sun_sensor_orientation = R.from_euler('xyz', SatellitePersonality.SUN_SENSOR_ORIENTATION, False)

        satellite_orientation_wrt_sensor = sun_sensor_orientation.inv().apply(satellite_orientation)

        return SensorData("Satellite orientation", satellite_orientation_wrt_sensor)

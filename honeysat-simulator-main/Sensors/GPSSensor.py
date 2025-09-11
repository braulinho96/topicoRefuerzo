from random import random, uniform
from typing import Callable

from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData

from Simulations.OrbitalSimulation import OrbitalSimulation


class GPSSensor(Sensor):
    """
    GPS SENSOR TO FEED GPS DATA TO THE FLIGHT SOFTWARE
    A sensor that returns lat, lon, and altitude based on the OrbitalSimulation.
    """

    def __init__(self, orbital_simulation: OrbitalSimulation):
        """
        Initialize the sensor.

        :param orbital_simulation: The OrbitalSimulation instance to use
        """

        super().__init__()

        self._sensor_name = "GPSSensor"

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be a OrbitalSimulation")
        self.orbital_simulation = orbital_simulation

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('altitude', self.measure_alt))
        list_of_methods.append(('latitude', self.measure_lat))
        list_of_methods.append(('longitude', self.measure_lon))

        # TODO add missing values

        return list_of_methods

    def measure_lat(self) -> SensorData:
        """
        Get the current simulated latitude measurement from the Orbital Simulation.

        Returns:
        - float: Simulated voltage reading.
        """
        lat, lon = self.orbital_simulation.send_request('get_lat_lon').result()['satellite_latitude'], [
            'satellite_longitude']

        return SensorData("Latitude", lat)

    def measure_lon(self) -> SensorData:
        """
        Get the current simulated longitude measurement from the Orbital Simulation.

        Returns:
        - float: Simulated voltage reading.
        """

        lat, lon = self.orbital_simulation.send_request('get_lat_lon').result()

        return SensorData("Longitude", lon)

    def measure_alt(self) -> SensorData:
        """
        Get the current simulated altitude measurement from the Orbital Simulation.

        Returns:
        - float: Simulated voltage reading.
        """

        alt = self.orbital_simulation.send_request('get_alt').result()

        return SensorData("Altitude", alt)

    def measure(self) -> SensorData:
        """
        Get the current simulated altitude measurement from the Orbital Simulation.

        Returns:
        - float: Simulated voltage reading.
        """

        result = self.orbital_simulation.send_request('get_gps_data').result()
        alt = result['satellite_altitude_degrees']
        lat = result['satellite_latitude']
        lon = result['satellite_longitude']

        return SensorData("Altitude", alt), SensorData("Latitude", lat), SensorData("Longitude", lon)

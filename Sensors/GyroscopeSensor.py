from typing import Callable

from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData

from Simulations.RotationSimulation import RotationSimulation


class GyroscopeSensor(Sensor):
    """
    Sensor that provides direction of rotation, angular velocity, rotation angle, and vibration
    """

    def __init__(self, rotation_simulation: RotationSimulation):
        """
        Initialize the sensor.

        :param rotation_simulation: The RotationSimulation instance to use
        """

        super().__init__()
        self._sensor_name = "GyroscopeSensor"

        if not rotation_simulation:
            raise ValueError("No rotation_simulation available")
        if not isinstance(rotation_simulation, RotationSimulation):
            raise TypeError("Simulation must be a RotationSimulation")

        self.rotation_simulation = rotation_simulation

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        list_of_methods = []
        list_of_methods.append(('gyroscope', self.get_all_data))

        return list_of_methods

    def get_all_data(self) -> SensorData:
        """
        After calling RotationSimulation, it returns the direction of rotation, angular velocity, rotation angle, and vibration

        :return: The direction of rotation, angular velocity, rotation angle, and vibration
        """

        data = self.rotation_simulation.send_request('all').result()

        return SensorData("Gyroscope", data)

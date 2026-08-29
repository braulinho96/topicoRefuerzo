from Sensors.AccelerometerSensor import AccelerometerSensor
from Sensors.SensorData import SensorData
from Simulations.MagneticSimulation import MagneticSimulation
from Simulations.RotationSimulation import RotationSimulation
from SubsystemsStates.SubsystemState import SubsystemState
from Sensors.Sensor import Sensor
import numpy as np


class ADCSState(SubsystemState):

    def __init__(self, sensors: list[Sensor], rotation_simulation: RotationSimulation,
                 magnetic_simulation: MagneticSimulation):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        super().__init__()
        self._subsystem_name = 'ADCS'

        # SENSORS CHECK AND STORE
        if not sensors:
            raise ValueError("No sensors available")
        for sensor in sensors:
            if not isinstance(sensor, Sensor):
                raise TypeError("Parameters must be a Sensor")

        self._sensors_list = sensors

        for sensor in self._sensors_list:
            sensor_available_data = sensor.get_sensor_list_of_methods()
            for single_data in sensor_available_data:
                self.available_sensor_data[single_data[0]] = single_data[1]
        # Uncomment the line below to print available sensor data methods
        # print(self.available_sensor_data.keys())

        # SIMULATIONS
        if not rotation_simulation or not isinstance(rotation_simulation, RotationSimulation):
            raise ValueError("No rotation simulation available")
        self._rotation_simulation = rotation_simulation

        if not magnetic_simulation or not isinstance(magnetic_simulation, MagneticSimulation):
            raise ValueError("No magnetic simulation available")
        self._magnetic_simulation = magnetic_simulation

    def receive_command(self, command: str, value) -> SensorData:
        match command:
            case "set_torque":
                if isinstance(value, np.ndarray) and value.shape == (3,):
                    self._rotation_simulation.set_torque(value)
                    return SensorData("status", True, bool)
                else:
                    raise ValueError("Invalid value for set_torque")
            case "set_torque_channel":
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    channel, new_torque = value[:]
                    torque = self._rotation_simulation.get_torque()
                    torque[channel] = new_torque
                    self._rotation_simulation.set_torque(torque)
                    return SensorData("status", True, bool)
                else:
                    raise ValueError("Invalid value for set_torque")
            case "update_dynamics":
                self._rotation_simulation.update_simulation()
                return SensorData("status", True, bool)

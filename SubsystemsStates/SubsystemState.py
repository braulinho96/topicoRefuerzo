from abc import ABC, abstractmethod

from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData


class SubsystemState(ABC):
    def __init__(self):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        self.available_sensor_data = {}
        self.available_settings = {}
        self._sensors_list = []
        self._subsystem_name = 'Abstract Subsystem'
        pass

    @abstractmethod
    def receive_command(self, command: str, value):
        """
        Abstract method to receive a command.
        This needs to be implemented in the derived classes.
        """
        if command in self.available_settings:
            return self.available_settings[command](value)
        pass

    def read_value(self, command: str) -> SensorData:
        """
        Generic method to receive a command.
        This could to be re-implemented in the derived classes.
        """
        if command in self.available_sensor_data:
            return self.available_sensor_data[command]()
        pass

    def get_subsystem_available_methods(self):
        if not self.available_sensor_data or self.available_sensor_data == {}:
            raise ValueError("No sensors available")
        if not isinstance(self.available_sensor_data, dict):
            raise TypeError("Parameters must be a dictionary")
        return self.available_sensor_data

    def get_subsystem_available_settings(self):
        if not self.available_settings or self.available_settings == {}:
            raise ValueError("No sensors settings available")
        if not isinstance(self.available_settings, dict):
            raise TypeError("Parameters must be a dictionary")
        return self.available_settings

    def print_subsystem_state(self):
        """
        Generic method to print the current details of the Subsystem state.
        This could to be re-implemented in the derived classes.
        """
        print(f"\t - {self._subsystem_name} subsystem\r")

        print(f"\t   The {self._subsystem_name} subsystem includes {len(self._sensors_list)} sensors:\r")

        for sensor in self._sensors_list:
            sensor.print_sensor_state()
            # print(f'\t\t{sensor}')



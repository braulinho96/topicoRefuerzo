from abc import ABC, abstractmethod
from typing import Callable
from Sensors.SensorData import SensorData


class Sensor(ABC):
    def __init__(self):
        """
        Initialize the sensor.
        """
        self._list_of_methods = self._generate_sensor_list_of_methods()
        self._sensor_name = "Abstract Sensor"
        pass

    @abstractmethod
    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        """
        The _generate_sensor_list_of_methods method generates a list of tuples containing method names and their corresponding
        functions. This method is abstract and must be implemented in the derived classes.
        """
        pass

    def get_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        """
        The get_list_of_methods method is useful for providing a structured way to access and iterate over a set of
        methods within a class. By returning a list of tuples containing method names and their corresponding
        functions, it allows for dynamic method invocation based on the method name. This approach is handy because
        we have multiple sensor or measurement functions within a class, and you want to abstract away the details of
        each measurement behind a common interface in SubsystemsStates.
        """
        return self._list_of_methods

    def print_sensor_state(self):
        """
        Generic method to print the current details of the sensor state.
        This could to be re-implemented in the derived classes.
        """
        print(f"\t\t- {self._sensor_name} sensor")
        print(f"\t\t  The {self._sensor_name} sensor can measure {len(self._list_of_methods)} data:\r")
        for single_data in self._list_of_methods:
            print(f"\t\t\t - {single_data[0]} data")
            # state_string += str(f"\t\t\tCurrent {single_data[1]().name} measure is: {single_data[1]().value}")

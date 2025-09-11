from abc import ABC, abstractmethod
from typing import Union
from Sensors.SensorData import SensorData
from SubsystemsStates.SubsystemState import SubsystemState
from SatellitePersonality import SatellitePersonality
from Interfaces import Interface


class SatState(ABC):
    def __init__(self, subsystems: list[SubsystemState], interface: Interface):
        """
        Initialize the satellite state. This method should be overridden by all subclasses.
        Pass a dictionary of keyword arguments containing the SubSystemStates required for the given Satellite.
        """
        self.subsystems = subsystems
        self.satellite_name = SatellitePersonality.SATELLITE_NAME
        self.interface = interface

        if not self.subsystems:
            raise ValueError("No subsystems available")
        for subsystem in self.subsystems:
            if not isinstance(subsystem, SubsystemState):
                raise ValueError("Parameters must be a SubsystemState")

        pass

    @abstractmethod
    def update(self, dt: Union[int, float]):
        """
        Abstract method to update the satellite state
        @param dt: update period
        """
        pass

    @abstractmethod
    def receive_command(self, command: str):
        """
        Abstract method to receive a command.
        This needs to be implemented in the derived classes.
        """
        pass

    @abstractmethod
    def read_value(self, command: str) -> SensorData:
        """
        Abstract method to receive a command.
        This needs to be implemented in the derived classes.
        """

        pass

    @abstractmethod
    def process_request(self, message: bytes) -> bytes:

        pass

    def print_satellite_state(self):
        """
        Generic method to print the current details of the Satellite state.
        This could to be re-implemented in the derived classes.
        """
        print(f"The HoneySat API is currently simulating the {self.satellite_name} satellite\r")
        print(f"The simulation includes {len(self.subsystems)} subsystems:\r")

        for subsystem in self.subsystems:
            subsystem.print_subsystem_state()


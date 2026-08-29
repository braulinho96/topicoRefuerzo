from Sensors.Sensor import Sensor
from Sensors.CameraPayload import CameraPayload
from Sensors.LangmuirPayload import LangmuirPayload
from Sensors.SensorData import SensorData
from SubsystemsStates.SubsystemState import SubsystemState


class PayloadState(SubsystemState):

    def __init__(self, sensors: [Sensor]):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        super().__init__()

        # PAYLOADState Attributes
        self._subsystem_name = 'Payload'
        self._sensors_list = sensors

        if not sensors:
            raise ValueError("No payload available")
        for sensor in sensors:
            if not isinstance(sensor, CameraPayload) and not isinstance(sensor, LangmuirPayload):
                raise TypeError("Parameter must be a Sensor object")

        for sensor in sensors:
            sensor_available_method = sensor.get_sensor_list_of_methods()
            for sensor_method in sensor_available_method:
                self.available_sensor_data[sensor_method[0]] = sensor_method[1]

        for sensor in sensors:
            if hasattr(sensor, 'get_sensor_list_of_settings'):
                sensor_available_setting = sensor.get_sensor_list_of_settings()
                for sensor_setting in sensor_available_setting:
                    self.available_settings[sensor_setting[0]] = sensor_setting[1]

        print(self.available_sensor_data)

    def get_subsystem_available_methods(self):
        return self.available_sensor_data

    def get_subsystem_available_settings(self):
        return self.available_settings

    def receive_command(self, command: str, value):
        # change some camera setting, for example
        return super().receive_command(command, value)

    def read_value(self, command: str) -> SensorData:
        data = super().read_value(command)

        return data

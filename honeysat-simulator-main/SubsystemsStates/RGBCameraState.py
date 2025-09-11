from Sensors.CameraPayload import CameraPayload
from Sensors.SensorData import SensorData
from SubsystemsStates.SubsystemState import SubsystemState


class RGBCameraState(SubsystemState):

    def receive_command(self, command: str):
        # change some camera setting, for example
        pass


    def read_value(self, command: str) -> SensorData:
        pass

    def __init__(self, camera_payload: CameraPayload):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        super().__init__()

        if not camera_payload:
            raise ValueError("No camera_payload available")
        if not isinstance(camera_payload, CameraPayload):
            raise TypeError("Simulation must be a CameraPayload")

        self.camera_payload = camera_payload
        self.available_sensor_data = self.camera_payload.get_sensor_list_of_methods()
        print(self.available_sensor_data)
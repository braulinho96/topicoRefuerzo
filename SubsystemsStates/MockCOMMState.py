from SubsystemsStates.SubsystemState import SubsystemState
from Sensors.SensorData import SensorData

class MockCOMMState(SubsystemState):
    """
    A mock COMMState that bypasses the multiprocessing hub to avoid pickling errors.
    It provides the same interface as the real COMMState but does nothing.
    It inherits from SubsystemState to satisfy the type check in SatState.
    """
    def __init__(self, sensors):
        super().__init__()  # Call the parent constructor
        self._sensors = sensors
        print("Initialized MockCOMMState, multiprocessing hub is bypassed.")

    def update(self, period):
        # In a real scenario, this would update sensor readings.
        # For the mock version, we can just pass.
        pass

    def get_state(self):
        # Return a mock state dictionary.
        return {
            "comm_mock_status": "OK"
        }

    def get_sensor(self, sensor_name):
        for sensor in self._sensors:
            if sensor.__class__.__name__ == sensor_name:
                return sensor
        return None

    def read_value(self, command: str) -> SensorData:
        """
        Mock implementation for reading a value.
        Specifically handles the 'is_comm_reachable' check from SuchaiState.
        """
        if command == 'is_comm_reachable':
            # Always return True for the mock state to allow the simulation to proceed.
            return SensorData(command, True)
        return SensorData(command, None)

    def receive_command(self, command: str, value):
        """
        Mock implementation for receiving a command. Does nothing.
        """
        pass
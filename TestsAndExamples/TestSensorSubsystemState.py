from SubsystemsStates.SubsystemState import SubsystemState
from TestsAndExamples.TestSensor import TestSensor
from TestsAndExamples.TestSimulation import TestSimulation


class TestSensorSubsystemState(SubsystemState):
    def __init__(self, test_simulation: TestSimulation):
        super().__init__()
        self.test_simulation = test_simulation
        self.test_sensor = TestSensor(test_simulation)

    def receive_command(self, command: str):
        """
        Receive a command and act accordingly.

        :param command: The command to execute
        """
        if command == "get_value":
            print(self.test_sensor.get_value().value)
        else:
            print(f"Unknown command: {command}")

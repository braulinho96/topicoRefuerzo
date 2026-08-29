import time
from random import random
from Sensors.Sensor import Sensor
from TestSimulation import TestSimulation
from Sensors.SensorData import SensorData


class TestSensor(Sensor):
    def __init__(self, test_simulation: TestSimulation):
        """
        Initialize the sensor.

        :param test_simulation: The TestSimulation instance to use
        """
        super().__init__()
        self.test_simulation = test_simulation

    def get_value(self) -> SensorData:
        """
        Retrieves the count values from the TestSimulation instance.

        :return: Sensordata object with the sum of the count values
        """

        if not self.test_simulation:
            raise ValueError("No test_simulation available")

        if not isinstance(self.test_simulation, TestSimulation):
            raise TypeError("Simulation must be a TestSimulation")

        sum_value = 0

        for i in range(5):
            try:
                # Send a request and immediately wait for the response
                response = self.test_simulation.send_request("Request " + str(i)).result(
                    timeout=10)  # Adjust timeout as needed

                # The response is a str and will end with a integer, split the string after "count is now " and
                # + add the integer to sum_value
                sum_value += int(response.split("count is now ")[1])

                print(response)
            except Exception as e:
                print(f"Exception occurred: {e}")

            # Wait a random amount of time before sending the next request
            time.sleep(1 + random() * 5)

        return SensorData("Sum", sum_value)

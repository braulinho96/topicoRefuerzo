import unittest

from Sensors.EveryFaceSunSensor import EveryFaceSunSensor
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.RotationSimulation import RotationSimulation


class EveryFaceSunSensorTests(unittest.TestCase):
    def setUp(self):
        # Initialize the simulations
        self.rotation_simulation = RotationSimulation()
        self.orbit_simulation = OrbitalSimulation(self.rotation_simulation)

        # Start the simulations
        self.rotation_simulation.start()
        self.orbit_simulation.start()

    def test_every_face_sun_sensor(self):
        sun_sensor = EveryFaceSunSensor(self.orbit_simulation)

        is_sun_present = sun_sensor.is_satellite_in_sunlight()
        satellite_orientation = sun_sensor.sun_orientation_wrt_satellite()
        single_sensor_satellite_orientation = sun_sensor.sun_orientation_wrt_satellite(5)

        # Check if the values are as expected
        self.assertIsNotNone(is_sun_present.value)
        self.assertIsNotNone(satellite_orientation.value)
        self.assertIsNotNone(single_sensor_satellite_orientation.value)

        # Optional: add more specific assertions here based on expected behavior

    def tearDown(self):
        # Stop the simulations
        self.orbit_simulation.stop()
        self.rotation_simulation.stop()


if __name__ == '__main__':
    unittest.main()

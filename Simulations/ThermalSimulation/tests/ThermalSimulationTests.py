import unittest
from unittest.mock import patch, Mock

from Simulations.ThermalSimulation.ThermalSimulation import ThermalSimulation

time_mock = Mock()
time_mock.return_value = 100


class ThermalSimTests(unittest.TestCase):

    @patch('time.time', time_mock)
    def test_simple_temp(self):
        osim = Mock()
        osim.get_orbital_is_sunlit.return_value = True
        tsim = ThermalSimulation(osim)
        for i in range(100):
            osim.get_orbital_is_sunlit.return_value = True
            for i in range(60*45):
                time_mock.return_value += 1
                tsim.update_simulation()
            osim.get_orbital_is_sunlit.return_value = False
            for i in range(60*45):
                time_mock.return_value += 1
                tsim.update_simulation()
        a = tsim.get_temperature()
        tsim = ThermalSimulation(osim)
        for i in range(1000):
            osim.get_orbital_is_sunlit.return_value = True
            for i in range(60*45):
                time_mock.return_value += 1
                tsim.update_simulation()
            osim.get_orbital_is_sunlit.return_value = False
            for i in range(60*45):
                time_mock.return_value += 1
                tsim.update_simulation()
        b = tsim.get_temperature()
        print(b)
        self.assertAlmostEqual(a,b)


if __name__ == '__main__':
    unittest.main()

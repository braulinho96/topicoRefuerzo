import time
import unittest
from unittest.mock import Mock, patch, MagicMock

from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.PowerSystemSimulation.CellArrangement import SingleCell, ParallelCellArrangement, SeriesCellArrangement
from Simulations.PowerSystemSimulation.ChargingStrategy import VoltageCutoffStrategy
from Simulations.PowerSystemSimulation.PowerSystemSimulation import PowerSystemSimulation
from Simulations.PowerSystemSimulation.SolarPowerEstimator import ConstantPowerEstimator
from Simulations.RotationSimulation import RotationSimulation

time_mock = Mock()
time_mock.return_value = 100


class PowerSimulationTests(unittest.TestCase):
    @patch('time.time', time_mock)
    def test_basic(self):
        osim = Mock()
        osim.get_orbital_is_sunlit.return_value = True

        psim = PowerSystemSimulation(pack_arrangement=SingleCell(), sp_estimator=ConstantPowerEstimator(1, osim),
                                     charging_strategy=VoltageCutoffStrategy(0.1, 4.1))
        psim.start()
        psim.set_power_consumption(10)
        b = psim.get_current()
        psim.stop()
        self.assertTrue(2.2 < b < 2.3)

    @patch('time.time', time_mock)
    def test_full_discharge(self):
        osim = Mock()
        osim.get_orbital_is_sunlit.return_value = False

        psim = PowerSystemSimulation(pack_arrangement=SingleCell(), sp_estimator=ConstantPowerEstimator(10, osim),
                                     charging_strategy=VoltageCutoffStrategy(0.1, 4.0))

        psim.set_power_consumption(10)
        for i in range(50):
            psim._run_simulation()
            time_mock.return_value += 100
            print(psim.get_voltage())
        psim.set_power_consumption(0)
        b = psim.get_voltage()
        self.assertTrue(3.5 > b > 3.0)

    @patch('time.time', time_mock)
    def test_full_charge(self):
        osim = Mock()
        osim.get_orbital_is_sunlit.return_value = True

        psim = PowerSystemSimulation(pack_arrangement=SingleCell(), sp_estimator=ConstantPowerEstimator(10, osim),
                                     charging_strategy=VoltageCutoffStrategy(0.1, 4.0), starting_soc=0.1)

        for i in range(100):
            psim._run_simulation()
            time_mock.return_value += 100
            print(psim.get_voltage())
        b = psim.get_voltage()
        self.assertTrue(4.11 > b > 3.9)


if __name__ == '__main__':
    unittest.main()

import time
import unittest
from unittest.mock import patch, Mock

from Simulations.PowerSystemSimulation.CellArrangement import SingleCell, ParallelCellArrangement, SeriesCellArrangement
from Simulations.PowerSystemSimulation.LiIonBatteryCell import LiIonBatteryCell

time_mock = Mock()
time_mock.return_value = 100


class BatteryCellTests(unittest.TestCase):

    @patch('time.time', time_mock)
    def test_simple_soc(self):
        bat = LiIonBatteryCell(capacity=3.5)
        self.assertEqual(bat.get_state_of_charge(), 1)

    @patch('time.time', time_mock)
    def test_simple_voltage(self):
        bat = LiIonBatteryCell(capacity=3.5, starting_soc=0.001)
        time_mock.return_value += 3600
        self.assertTrue(bat.get_voltage() < 2.6)
        self.assertTrue(bat.get_state_of_charge() < 0.01)

    @patch('time.time', time_mock)
    def test_charge_discharge(self):
        bat = LiIonBatteryCell(capacity=3.5)
        bat.set_current(1)
        time_mock.return_value += 3600
        bat.set_current(-1)
        time_mock.return_value += 3600
        bat.set_current(0)
        time_mock.return_value += 3600
        self.assertTrue(bat.get_voltage() > 4)

    @patch('time.time', time_mock)
    def test_simulation_flattening(self):
        bat = LiIonBatteryCell(capacity=3.5)
        for i in range(5):
            bat.set_current(0.2)
            time_mock.return_value += 10
            bat.set_current(0.1)
            time_mock.return_value += 10
            bat.set_current(-0.2)
            time_mock.return_value += 10
            bat.set_current(-0.1)
            time_mock.return_value += 10
        bat.set_current(0)
        print(bat.get_voltage())

    @patch('time.time', time_mock)
    def test_lazy_precompute(self):
        bat = LiIonBatteryCell(capacity=3.0, steps_before_flatten=5)
        bat.set_current(2.0)
        for i in range(20):
            time_mock.return_value += 100
        bat.set_current(0.0)
        print(bat.get_voltage())


class CellArrangementTests(unittest.TestCase):
    def test_simple(self):
        cell = SingleCell()
        self.assertEqual(cell.get_pack_voltage(4.1), 4.1)

    def test_parallel(self):
        cell = ParallelCellArrangement(SingleCell(), 4)
        self.assertEqual(cell.get_cell_current(12), 12 / 4)

    def test_series(self):
        cell = SeriesCellArrangement(SingleCell(), 4)
        self.assertEqual(cell.get_pack_voltage(4.1), 4.1 * 4)


if __name__ == '__main__':
    unittest.main()

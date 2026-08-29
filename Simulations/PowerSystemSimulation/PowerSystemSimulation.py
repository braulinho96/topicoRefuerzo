from datetime import datetime, timedelta
from queue import Empty

from SatellitePersonality import SatellitePersonality
from Simulations.PowerSystemSimulation.CellArrangement import CellArrangement
from Simulations.PowerSystemSimulation.ChargingStrategy import ChargingStrategy
from Simulations.PowerSystemSimulation.SolarPowerEstimator import SolarPowerEstimator
from Simulations.Simulation import Simulation

from Simulations.PowerSystemSimulation.LiIonBatteryCell import LiIonBatteryCell

# temporal fix
from Simulations.PowerSystemSimulation.MockBatteryCell import MockBatteryCell


class PowerSystemSimulation(Simulation):
    __reference_cell: LiIonBatteryCell
    __pack_arrangement: CellArrangement
    __sp_estimator: SolarPowerEstimator
    __charging_strategy: ChargingStrategy

    __saved_cell_voltage: float
    __state_of_charge: float
    __input_power: float
    __output_power: float
    __idle_power: float

    def __init__(self, pack_arrangement: CellArrangement, sp_estimator: SolarPowerEstimator,
                 charging_strategy: ChargingStrategy, idle_power: float = 0.0, starting_soc=0.8):
        super().__init__()
        #self.__reference_cell = LiIonBatteryCell(capacity=SatellitePersonality.NOMINAL_CAPACITY, starting_soc=starting_soc)
        self.__reference_cell = MockBatteryCell(capacity=SatellitePersonality.NOMINAL_CAPACITY, starting_soc=starting_soc)
        self.__pack_arrangement = pack_arrangement
        self.__sp_estimator = sp_estimator
        self.__charging_strategy = charging_strategy
        self.__saved_cell_voltage = SatellitePersonality.NOMINAL_CAPACITY  # self.__reference_cell.get_voltage()
        self.__output_power = idle_power
        self.__input_power = sp_estimator.get_power()
        self.__state_of_charge = starting_soc

    def _run_simulation(self):

        try:
            request = self._request_queue.get(timeout=0.1)

            # TODO write a list of the request that this simulation can receive
            # For example, use "get attribute" to get the name of the function. If too much time use switch.

            match request.data:
                case "get_voltage":
                    voltage = self.get_voltage()
                    result = {
                        'voltage': voltage,
                    }
                case "get_current":
                    current = self.get_current()
                    result = {
                        'current': current,
                    }

                case "get_power_out":
                    power_out = self.get_power_out()
                    result = {
                        'power_out': power_out,
                    }

                case "get_power_in":
                    power_in = self.get_power_in()
                    result = {
                        'power_in': power_in,
                    }

                case "get_state_of_charge":
                    state_of_charge = self.get_state_of_charge()
                    result = {
                        'state_of_charge': state_of_charge,
                    }

                case "get_state_of_charge":
                    state_of_charge = self.get_state_of_charge()
                    result = {
                        'state_of_charge': state_of_charge,
                    }

                case default:
                    result = {

                    }

            if request is None:
                return

        except Empty:
            # current drawn by subsystems

            self.__input_power = self.__sp_estimator.get_power()

            # per cell current output
            output_current = self.__pack_arrangement.get_cell_current(
                self.__output_power / self.__pack_arrangement.get_pack_voltage(
                    self.__reference_cell.get_nominal_voltage()))

            # per cell current input
            input_current = self.__pack_arrangement.get_cell_current(
                self.__input_power / self.__pack_arrangement.get_pack_voltage(
                    self.__reference_cell.get_nominal_voltage()))

            cell_current = self.__charging_strategy.get_battery_current(self.__reference_cell.get_voltage(),
                                                                        output_current,
                                                                        input_current,
                                                                        self.__reference_cell.get_state_of_charge())
            self.__reference_cell.set_current(cell_current)

            self.__saved_cell_voltage = self.__reference_cell.get_voltage()

            self.__state_of_charge = self.__reference_cell.get_state_of_charge()

        except Exception as e:
            # If an exception occurs, set it on the future
            if request and request.id in self._futures:
                self._set_exception(request.id, e)
            else:
                print(f"Exception in simulation: {e}")

    def get_voltage(self):
        return self.__pack_arrangement.get_pack_voltage(self.__saved_cell_voltage)

    def get_current(self):
        return (self.__output_power - self.__input_power) / self.get_voltage()

    def get_power_out(self):
        return self.__output_power

    def get_power_in(self):
        return self.__input_power

    def get_state_of_charge(self):
        return self.__state_of_charge

    def set_power_consumption(self, power: float):
        if power < 0:
            raise ValueError("Power consumption must be positive")
        self.__output_power = power

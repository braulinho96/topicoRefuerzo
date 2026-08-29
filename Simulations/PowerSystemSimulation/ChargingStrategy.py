from abc import ABC, abstractmethod


class ChargingStrategy(ABC):
    @abstractmethod
    def get_battery_current(self, voltage: float, output_current: float, input_current: float,
                            state_of_charge: float) -> float:
        pass


class VoltageCutoffStrategy(ChargingStrategy):
    deadband: float
    setpoint: float
    __is_charging: bool

    def __init__(self, deadband: float, setpoint: float):
        self.deadband = deadband
        self.setpoint = setpoint
        self.__is_charging = False

    def get_battery_current(self, voltage: float, output_current: float, input_current: float,
                            state_of_charge: float) -> float:
        """
        Decide the charging current of a BatteryCell.
        :param state_of_charge:
        :param input_current:
        :param output_current:
        :param voltage: voltage of the battery
        """
        if voltage > self.setpoint + self.deadband:  # battery assumed to be fully charged
            self.__is_charging = False
        elif voltage <= self.setpoint:  # battery requires charging as it is below the setpoint
            self.__is_charging = True

        if self.__is_charging:
            return output_current - input_current
        else:
            return max(0.0, output_current - input_current)  # stop charging when fully charged

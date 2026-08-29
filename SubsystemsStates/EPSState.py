from enum import Enum

from SatellitePersonality import SatellitePersonality
from Sensors.Sensor import Sensor
from Sensors.SensorData import SensorData
from Sensors.VoltageSensor import VoltageSensor
from Simulations.PowerSystemSimulation.PowerSystemSimulation import PowerSystemSimulation
from SubsystemsStates.SubsystemState import SubsystemState


class EPSState(SubsystemState):
    __power_system_simulation: PowerSystemSimulation

    __heater_active: bool
    __heater_power: float  # Watt
    __gpio_output_state: list[bool]  # 6 channels
    __gpio_power_draws: list[float]  # power in Watt

    def __init__(self, sensors: list[Sensor], gpio_power_draws: list[float],
                 power_system_simulation: PowerSystemSimulation):
        """
        Initialize the EPS state. The constructor is overridden from the SubsystemState.
        """
        super().__init__()
        self._subsystem_name = 'EPS'
        self._sensors_list = sensors

        if len(gpio_power_draws) != 6:
            raise ValueError("Invalid number of GPIO")

        self.__gpio_power_draws = gpio_power_draws
        self.__gpio_output_state = [False, False, False, False, False, False]
        self.__heater_active = False
        self.__heater_power = SatellitePersonality.HEATER_POWER
        self.__power_system_simulation = power_system_simulation

        # SENSORS CHECK AND STORE
        if not sensors:
            raise ValueError("No sensors available")
        for sensor in sensors:
            if not isinstance(sensor, Sensor):
                raise TypeError("Parameters must be a Sensor")

        # FILL LIST OF AVAILABLE SENSOR DATA (DICT IS DECLARED IN SUPER)
        for sensor in sensors:
            sensor_available_data = sensor.get_sensor_list_of_methods()
            for single_data in sensor_available_data:
                self.available_sensor_data[single_data[0]] = single_data[1]

        # FILL LIST OF AVAILABLE SENSOR SETTINGS (DICT IS DECLARED IN SUPER)
        self.list_of_settable_methods = []
        self.list_of_settable_methods.append(('set_heater', self.set_heater))
        self.list_of_settable_methods.append(('set_output', self.set_output))
        self.list_of_settable_methods.append(('hard_reset', self.hard_reset))

        available_settings = self.list_of_settable_methods
        for single_data in available_settings:
            self.available_settings[single_data[0]] = single_data[1]

    def receive_command(self, command: str, value):
        """
        Method to receive a command.
        """
        return super().receive_command(command, value)

    def __update_power_consumption(self):
        total_power_draw = sum([j for i, j in zip(self.__gpio_output_state, self.__gpio_power_draws) if i])

        if self.__heater_active:
            total_power_draw += self.__heater_power

        self.__power_system_simulation.set_power_consumption(total_power_draw)

    def set_heater(self, enabled: bool):
        self.__heater_active = enabled
        self.__update_power_consumption()
        return SensorData(data_type=bool, value=enabled, name="enabled")

    def set_output(self, channel: int, enabled: bool):
        if channel < 0 or channel > 5:
            raise ValueError("Invalid channel")

        self.__gpio_output_state[channel] = enabled
        self.__update_power_consumption()

    def hard_reset(self, value: bool):
        """
        Process a hard-reset command. Nothing to do, just disable som consumptions
        :param value: Not used
        :return:
        """
        self.__heater_active = False
        self.__update_power_consumption()
        return SensorData(data_type=bool, value=True, name="status")

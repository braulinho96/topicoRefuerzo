import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pybamm
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Any


class BatteryCell(ABC):
    @abstractmethod
    def get_voltage(self):
        pass

    @abstractmethod
    def get_current(self):
        pass

    @abstractmethod
    def get_state_of_charge(self):
        pass

    @abstractmethod
    def set_current(self, current):
        pass

    @abstractmethod
    def get_nominal_voltage(self):
        pass

    @abstractmethod
    def cutoff_voltage(self):
        pass


@dataclass
class SimulationStep:
    step: Any
    soc: float
    end_time: time  # wall clock time at which the step ends
    start_time: time  # wall clock time at which the step starts


@dataclass
class StoredValues:
    time: list[float]  # wall clock time at which the voltage is valid
    voltage: list[float]
    capacity: list[float]

    


def background_worker(steps, soc, parameter_values):
    pybamm.set_logging_level("ERROR")
    experiment = pybamm.Experiment(steps)
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
    sim.solve(initial_soc=soc)
    times = sim.solution["Time [s]"].entries
    capacity = sim.solution["Discharge capacity [A.h]"].entries
    voltage = sim.solution["Voltage [V]"].entries
    return times, voltage, capacity


class LiIonBatteryCell(BatteryCell):
    __simulation_start: time
    __steps: list[SimulationStep]  # list of steps to be performed until the last current change.
    __values: StoredValues
    __current: float  # battery current in A (positive for discharge, negative for charge)
    __capacity: float  # battery capacity in Ah
    __parameter_values: pybamm.ParameterValues  # pybamm cell parameters
    __starting_soc: float  # initial state of charge
    __steps_before_flatten: int  # number of steps before the simulation is flattened
    __lookahead_time: int  # time in seconds to look ahead for the simulation
    __initial_lookahead: int  # initial time in seconds to look ahead for the simulation

    def __init__(self, starting_soc=1, capacity=3.5, parameter_values=None,
                 steps_before_flatten=5):
        """
        Create a Lithium-ion battery cell model
        :param capacity: nominal cell capacity in Ah
        :param parameter_values: pybamm parameter values
        :param starting_soc: initial state of charge
        """
        if parameter_values is None:
            parameter_values = pybamm.ParameterValues("Chen2020")
        self.__simulation_start = time.time()
        self.__steps = []
        self.__values = StoredValues([], [], [])
        self.__current = 0
        self.__capacity = capacity
        self.__parameter_values = parameter_values
        self.__parameter_values["Nominal cell capacity [A.h]"] = capacity
        self.__starting_soc = starting_soc
        self.__steps_before_flatten = steps_before_flatten
        self.__initial_lookahead = 3600
        self.__lookahead_time = self.__initial_lookahead

    def get_voltage(self):
        """
        Get the voltage of the Cell
        :return: Voltage in V
        """
        if not self.__values.time or time.time() > self.__values.time[-1]:
            self.__compute_simulation()
        return np.interp(time.time(), self.__values.time, self.__values.voltage)

    def get_current(self):
        """
        Get the current currently drawn from the Cell
        :return: current in A
        """
        return self.__current

    def get_state_of_charge(self) -> float:
        if not self.__values.time:
            return self.__starting_soc
        if not self.__values.time or time.time() > self.__values.time[-1]:
            self.__compute_simulation()
        # compute capacity loss since last current change
        cap_loss = float(np.interp(time.time(), self.__values.time, self.__values.capacity))
        # add capacity loss to initial state of charge of the current step
        soc = self.__starting_soc
        if self.__steps:
            soc = self.__steps[-1].soc
        return soc - cap_loss / self.__capacity

    def set_current(self, current):
        """
        Set the current currently drawn from the battery
        :param current: current in A
        :return:
        """
        if current == self.__current:
            return
        timestamp = time.time()

        self.__lookahead_time = self.__initial_lookahead

        previous_step_end = self.__simulation_start
        if self.__steps:
            previous_step_end = self.__steps[-1].end_time
        old_current = self.__current
        self.__current = current
        self.__steps.append(SimulationStep(pybamm.current(old_current,
                                                               duration=f'{math.floor(timestamp - previous_step_end) + 1} second',
                                                               period="1 second"), self.get_state_of_charge(),
                                           timestamp, previous_step_end))
        self.__compute_simulation()

    def get_nominal_voltage(self):
        return 3.7

    def cutoff_voltage(self):
        return 3.0

    def __compute_simulation(self):
        with (ProcessPoolExecutor() as ex):
            steps = [step.step for step in self.__steps] + [
                pybamm.current(self.__current, duration=f'{self.__lookahead_time} seconds', period="1 second")]
            self.__steps = self.__steps[-self.__steps_before_flatten:]
            self.__lookahead_time = self.__lookahead_time * 2
            initial_soc = self.__starting_soc
            if self.__steps:
                initial_soc = self.__steps[0].soc
            future = ex.submit(background_worker, steps, initial_soc, self.__parameter_values)
            res = future.result()
            start_time = time.time()
            if self.__steps:
                start_time = self.__steps[0].start_time
            self.__values.time = list(map(lambda x: x + start_time, res[0]))
            self.__values.voltage = res[1]
            self.__values.capacity = res[2]

import random
import time

import numpy as np

from Interfaces.ZMQInterface import ZMQInterface
from SatStates.SuchaiState import SuchaiState
from Sensors.AccelerometerSensor import AccelerometerSensor
from Sensors.GStoFSCommSensor import GStoFSCommSensor
from Sensors.GyroscopeSensor import GyroscopeSensor
from Sensors.MagnetometerSensor import MagnetometerSensor
from Sensors.SunSensor import SunSensor
from Sensors.CameraPayload import CameraPayload
from Simulations.MagneticSimulation import MagneticSimulation
from Simulations.RotationSimulation import RotationSimulation
from SubsystemsStates.ADCSState import ADCSState
from SubsystemsStates.EPSState import EPSState
from SubsystemsStates.PayloadState import PayloadState
from TestSimulation import TestSimulation
from TestSensor import TestSensor
from Sensors.TemperatureSensor import TemperatureSensor
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.PowerSystemSimulation import PowerSystemSimulation
from TestSensorSubsystemState import TestSensorSubsystemState
from Sensors.GPSSensor import GPSSensor
from Sensors.CurrentSensor import CurrentSensor
from Sensors.VoltageSensor import VoltageSensor
from SubsystemsStates.COMMState import COMMState

# POWER SYSTEM SIMULATION IMPORTS
from Simulations.PowerSystemSimulation.CellArrangement import SingleCell, SeriesCellArrangement, ParallelCellArrangement
from Simulations.PowerSystemSimulation.ChargingStrategy import VoltageCutoffStrategy
from Simulations.PowerSystemSimulation.PowerSystemSimulation import PowerSystemSimulation
from Simulations.PowerSystemSimulation.SolarPowerEstimator import ConstantPowerEstimator

# THERMAL SIMULATION IMPORTS
from Simulations.ThermalSimulation.ThermalSimulation import ThermalSimulation


def test_zmqinterface():
    rotation_simulation = RotationSimulation()
    rotation_simulation.start()

    orbital_simulation = OrbitalSimulation(rotation_simulation)
    orbital_simulation.start()

    magnetic_simulation = MagneticSimulation(orbital_simulation, rotation_simulation)
    magnetic_simulation.start()

    # we have 6 battery cells in total and a voltage of no more than 8.4V fully charged

    power_system_simulation = PowerSystemSimulation(
        pack_arrangement=ParallelCellArrangement(SeriesCellArrangement(SingleCell(), 2), 3),
        sp_estimator=ConstantPowerEstimator(9.7, orbital_simulation),
        charging_strategy=VoltageCutoffStrategy(0.1, 4.0),
        idle_power=0.6,)

    power_system_simulation.start()

    thermal_simulation = ThermalSimulation(orbital_simulation)
    thermal_simulation.start()

    sensors = [
        AccelerometerSensor(orbital_simulation),
        GyroscopeSensor(rotation_simulation),
        SunSensor(orbital_simulation),
        GPSSensor(orbital_simulation),
        MagnetometerSensor(magnetic_simulation, rotation_simulation),
    ]

    adcs_state = ADCSState(sensors, rotation_simulation, magnetic_simulation)

    comm_sensors = [
        GStoFSCommSensor(orbital_simulation)
    ]

    comm_state = COMMState(comm_sensors)

    eps_sensors = [
        TemperatureSensor(thermal_simulation),
        VoltageSensor(power_system_simulation),
        CurrentSensor(power_system_simulation)
    ]

    # They reflect the additional power that is consumed if the corresponding output is enabled in Watt
    # These are meant for the power draw of other subsystems.
    power_draws = [
        0.1,  # 2 to 5 Watts
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    eps_state = EPSState(eps_sensors, power_draws, power_system_simulation)

    try:
        camera = CameraPayload(orbital_simulation)
        payload_state = PayloadState([camera])
        subsystem_states = {
            'eps': eps_state,
            'comm': comm_state,
            'adcs': adcs_state,
            'payload': payload_state
        }
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"An error occurred while initializing the camera: {e}")
        # Initialize subsystem_states without the 'payload'
        subsystem_states = {
            'eps': eps_state,
            'comm': comm_state,
            'adcs': adcs_state
        }

    zmq_interface = ZMQInterface()
    suchai_state = SuchaiState(subsystem_states, zmq_interface)
    suchai_state.print_satellite_state()

    try:
        period = 0.1
        while True:
            ti = time.time()
            suchai_state.update(period)
            dt = period - (time.time() - ti)
            # print(f"dt value is {dt}")
            # assert (dt > 0)
            time.sleep(dt if dt > 0 else 0)
    except KeyboardInterrupt:
        pass

    # power_system_simulation.stop()
    rotation_simulation.stop()
    orbital_simulation.stop()


if __name__ == "__main__":
    test_zmqinterface()

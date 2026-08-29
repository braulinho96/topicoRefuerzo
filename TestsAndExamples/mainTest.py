import random
import time

import numpy as np
import os

from SatStates.SuchaiState import SuchaiState
from Sensors.AccelerometerSensor import AccelerometerSensor
from Sensors.CurrentSensor import CurrentSensor
from Sensors.EveryFaceSunSensor import EveryFaceSunSensor
from Sensors.GPSSensor import GPSSensor
from Sensors.GyroscopeSensor import GyroscopeSensor
from Sensors.LangmuirPayload import LangmuirPayload
from Sensors.CameraPayload import CameraPayload
from Simulations.ThermalSimulation.ThermalSimulation import ThermalSimulation
from SubsystemsStates.PayloadState import PayloadState
from Sensors.MagnetometerSensor import MagnetometerSensor
from Sensors.SunSensor import SunSensor
from Sensors.TemperatureSensor import TemperatureSensor
from Sensors.VoltageSensor import VoltageSensor
from Simulations.MagneticSimulation import MagneticSimulation
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.PowerSystemSimulation import PowerSystemSimulation
from Simulations.RotationSimulation import RotationSimulation
from SubsystemsStates.ADCSState import ADCSState
from SubsystemsStates.COMMState import COMMState
from SubsystemsStates.EPSState import EPSState
from TestSensorSubsystemState import TestSensorSubsystemState
from TestSimulation import TestSimulation


def test_TestSensor():
    # Initialize TestSimulation
    sim = TestSimulation()

    # Start the simulation
    sim.start()

    test_subsystem_state = TestSensorSubsystemState(sim)

    test_subsystem_state.receive_command("get_value")

    # Stop the simulation
    sim.stop()


def test_TemperatureSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulations
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    thermal_simulation = ThermalSimulation(orbit_simulation)
    thermal_simulation.start()

    temperature_sensor = TemperatureSensor(thermal_simulation)
    temperature = temperature_sensor.get_value()

    print(
        f"Temperature in Kelvin: {temperature.value:.2f} - Temperature in Celsius: {float(temperature.value) - 273.15:.2f}")
    thermal_simulation.stop()
    orbit_simulation.stop()
    rotation_simulation.stop()


def test_SunSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulations
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    sun_sensor = SunSensor(orbit_simulation)

    is_sun_present = sun_sensor.is_satellite_in_sunlight()
    satellite_orientation = sun_sensor.sun_orientation_wrt_satellite()

    print(f"Is sun present: {is_sun_present.value}")
    print(f"Satellite orientation: {satellite_orientation.value}")

    rotation_simulation.set_torque(np.array([0, 0, 0.002]))

    time.sleep(5)

    rotation_simulation.set_torque(np.array([0.0, 0.0, 0.0]))

    is_sun_present = sun_sensor.is_satellite_in_sunlight()
    sun_orientation = sun_sensor.sun_orientation_wrt_satellite()

    print(f"Is sun present: {is_sun_present.value}")
    print(f"Satellite orientation: {sun_orientation.value}")
    orbit_simulation.stop()

    rotation_simulation.stop()


def test_EveryFaceSunSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulations
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    sun_sensor = EveryFaceSunSensor(orbit_simulation)

    is_sun_present = sun_sensor.is_satellite_in_sunlight()
    satellite_orientation = sun_sensor.sun_orientation_wrt_satellite()
    single_sensor_satellite_orientation = sun_sensor.sun_orientation_wrt_satellite(5)
    angle_incidence = sun_sensor.sun_angle_incidence()
    single_angle_incidence = sun_sensor.sun_angle_incidence(5)
    angle_incidence_as_current = sun_sensor.sun_angle_incidence_as_current()
    single_angle_incidence_as_current = sun_sensor.sun_angle_incidence_as_current(5)

    print(f"Is sun present: {is_sun_present.value}")
    print(f"Satellite orientation: {satellite_orientation.value}")
    print(f"Single Satellite orientation: {single_sensor_satellite_orientation.value}")
    print(f"Angle incidence: {angle_incidence.value}")
    print(f"Single angle incidence: {single_angle_incidence.value}")
    print(f"Angle incidence as current: {angle_incidence_as_current.value}")
    print(f"Single angle incidence as current: {single_angle_incidence_as_current.value}")

    orbit_simulation.stop()

    rotation_simulation.stop()


def test_EveryFaceSunSensor_with_rotation():
    rotation_simulation = RotationSimulation()

    # Start the simulations
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    sun_sensor = EveryFaceSunSensor(orbit_simulation)

    print(sun_sensor.orbital_simulation.rotation_simulation.send_request("all").result())
    is_sun_present = sun_sensor.is_satellite_in_sunlight()
    satellite_orientation = sun_sensor.sun_orientation_wrt_satellite()
    angle_incidence = sun_sensor.sun_angle_incidence()
    angle_incidence_as_current = sun_sensor.sun_angle_incidence_as_current()

    print(f"Is sun present: {is_sun_present.value}")
    print(f"Satellite orientation: {satellite_orientation.value}")
    print(f"Angle incidence: {angle_incidence.value}")
    print(f"Angle incidence as current: {angle_incidence_as_current.value}")

    rotation_simulation.set_torque(np.array([-0.001, -0.0002, 0.002]))
    # sleep 2 seconds
    time.sleep(10)

    print(sun_sensor.orbital_simulation.rotation_simulation.send_request("all").result())

    is_sun_present = sun_sensor.is_satellite_in_sunlight()
    satellite_orientation = sun_sensor.sun_orientation_wrt_satellite()
    angle_incidence = sun_sensor.sun_angle_incidence()
    angle_incidence_as_current = sun_sensor.sun_angle_incidence_as_current()

    print(f"Is sun present: {is_sun_present.value}")
    print(f"Satellite orientation: {satellite_orientation.value}")
    print(f"Angle incidence: {angle_incidence.value}")
    print(f"Angle incidence as current: {angle_incidence_as_current.value}")

    orbit_simulation.stop()

    rotation_simulation.stop()


def test_GyroscopeSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    gyroscope_sensor = GyroscopeSensor(rotation_simulation)

    # print this 'angular_acceleration', 'angular_velocity', 'orientation', 'time', 'torque'
    rotation_simulation.set_torque(
        np.array([random.uniform(0, 0.003), random.uniform(0, 0.003), random.uniform(0, 0.003)]))
    data = gyroscope_sensor.get_all_data()
    print(
        f"Time: {data.value['time']} - Orientation: {data.value['orientation']} - Angular velocity: {data.value['angular_velocity']} - Angular acceleration: {data.value['angular_acceleration']} - Torque: {data.value['torque']}")

    time.sleep(5)

    rotation_simulation.set_torque(np.array([0.0, 0.0, 0.0]))

    data = gyroscope_sensor.get_all_data()

    # print this 'angular_acceleration', 'angular_velocity', 'orientation', 'time', 'torque'
    print(
        f"Time: {data.value['time']} - Orientation: {data.value['orientation']} - Angular velocity: {data.value['angular_velocity']} - Angular acceleration: {data.value['angular_acceleration']} - Torque: {data.value['torque']}")

    time.sleep(5)

    data = gyroscope_sensor.get_all_data()

    # print this 'angular_acceleration', 'angular_velocity', 'orientation', 'time', 'torque'
    print(
        f"Time: {data.value['time']} - Orientation: {data.value['orientation']} - Angular velocity: {data.value['angular_velocity']} - Angular acceleration: {data.value['angular_acceleration']} - Torque: {data.value['torque']}")

    rotation_simulation.stop()


def test_GPSSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    gps_sensor = GPSSensor(orbit_simulation)
    alt, lat, lon = gps_sensor.measure()

    print(
        f"Altitude: {alt.value} - Latitude: {lat.value} - Latitude: {lon.value}")
    orbit_simulation.stop()
    rotation_simulation.stop()


def test_MagneticSimulation():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    magnetic_simulation = MagneticSimulation(orbit_simulation, rotation_simulation)
    magnetic_simulation.start()

    magnetic_field = magnetic_simulation.send_request('earth_magnetic_field').result()
    print(magnetic_field)

    print(f"North component (nT): {magnetic_field['north']}")
    print(f"North component (mGauss): {magnetic_field['north'] * 0.01}")
    print(f"East component (nT): {magnetic_field['east']}")
    print(f"East component (mGauss): {magnetic_field['east'] * 0.01}")
    print(f"Vertical component (nT): {magnetic_field['vertical']}")
    print(f"Vertical component (mGauss): {magnetic_field['vertical'] * 0.01}")
    print(f"Total intensity (nT): {magnetic_field['total_intensity']}")
    print(f"Total intensity (mGauss): {magnetic_field['total_intensity'] * 0.01}")

    magnetic_simulation.stop()
    orbit_simulation.stop()
    rotation_simulation.stop()


def test_MagnetometerSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    magnetic_simulation = MagneticSimulation(orbit_simulation, rotation_simulation)
    magnetic_simulation.start()

    magnetometer_sensor = MagnetometerSensor(magnetic_simulation, rotation_simulation)

    magnetic_field_mGauss = magnetometer_sensor.get_magnetic_field_mGauss()

    print(f"Magnetic field in mGauss: {magnetic_field_mGauss.value}")

    magnetic_simulation.stop()
    orbit_simulation.stop()
    rotation_simulation.stop()


def test_CurrentSensor():
    power_simulation = PowerSystemSimulation()
    power_simulation.start()

    current_sensor = CurrentSensor(power_simulation)
    current = current_sensor.measure_current()

    print(
        f"Current: {current.value:.2f}")
    power_simulation.stop()


def test_VoltageSensor():
    power_simulation = PowerSystemSimulation()
    power_simulation.start()

    voltage_sensor = VoltageSensor(power_simulation)
    current = voltage_sensor.measure_voltage()

    print(
        f"Voltage: {current.value:.2f}")
    power_simulation.stop()


def test_AccelerometerSensor():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    accelerometer = AccelerometerSensor(orbit_simulation)

    accelerometer_data = accelerometer.get_acceleration()
    print(f"Acceleration: {accelerometer_data.value}")

    # rotation_simulation.set_torque(np.array([0, 0, 0.002]))
    #
    time.sleep(10)
    #
    # rotation_simulation.set_torque(np.array([0.0, 0.0, 0.0]))

    accelerometer_data = accelerometer.get_acceleration()
    print(f"Acceleration: {accelerometer_data.value}")

    orbit_simulation.stop()

    rotation_simulation.stop()


def test_COMMState():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    comm_state = COMMState(orbit_simulation)
    print(comm_state.is_comm_reachable())

    orbit_simulation.stop()
    rotation_simulation.stop()


def test_ADCSState():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    sensors = [
        AccelerometerSensor(orbit_simulation),
        GyroscopeSensor(rotation_simulation),
        SunSensor(orbit_simulation),
        GPSSensor(orbit_simulation)
    ]

    # accelerometer_data = accelerometer.get_acceleration()
    # print(f"Acceleration: {accelerometer_data.value}")
    #
    # accelerometer_data = accelerometer.get_acceleration()
    # print(f"Acceleration: {accelerometer_data.value}")

    adcs_state = ADCSState(sensors)

    # adcs_state.print_subsystem_state()
    orbit_simulation.stop()

    rotation_simulation.stop()


def test_EPSState():
    power_system_simulation = PowerSystemSimulation()

    power_system_simulation.start()

    voltage = VoltageSensor(power_system_simulation)

    eps_state = EPSState([voltage])

    method_name = 'voltage'
    command = eps_state.read_value(method_name)
    print(command.value)

    power_system_simulation.stop()


def test_SuchaiState():
    rotation_simulation = RotationSimulation()

    # Start the simulation
    rotation_simulation.start()

    orbit_simulation = OrbitalSimulation(rotation_simulation)
    orbit_simulation.start()

    comm_state = COMMState(orbit_simulation)

    power_system_simulation = PowerSystemSimulation()

    power_system_simulation.start()

    voltage = VoltageSensor(power_system_simulation)

    eps_state = EPSState([voltage])

    subsystem_states = {
        'eps_state': eps_state,
        'comm_state': comm_state,
    }

    suchai_state = SuchaiState(**subsystem_states)
    suchai_state.print_satellite_state()

    power_system_simulation.stop()
    orbit_simulation.stop()
    rotation_simulation.stop()


def test_LangmuirPayload():
    rot = RotationSimulation()
    rot.start()

    orb = OrbitalSimulation(rot)
    orb.start()

    langmuir_payload = LangmuirPayload(orb)

    data = langmuir_payload.get_value()
    print(data.value)

    time.sleep(1)

    data = langmuir_payload.get_value()
    print(data.value)

    orb.stop()
    rot.stop()


def test_CameraPayload():
    rotation_simulation = RotationSimulation()
    rotation_simulation.start()

    orb_simulation = OrbitalSimulation(rotation_simulation)
    orb_simulation.start()

    camera = CameraPayload(orb_simulation)

    data = camera.get_image()
    print("Selected Image Path: {}".format(data.value))

    status = camera.set_image_size(256, 236)

    print("Command Status for Resizing Image: {}".format(status.value))

    data = camera.get_image()

    assert os.path.exists(data.value)
    print("Selected Image Path: {}".format(data.value))

    orb_simulation.stop()
    rotation_simulation.stop()


def test_PayloadState():
    rot = RotationSimulation()
    rot.start()

    orb = OrbitalSimulation(rot)
    orb.start()

    langmuir = LangmuirPayload(orb)
    camera = CameraPayload(orb)

    sensors = [
        langmuir,
        camera
    ]

    payload = PayloadState(sensors)

    print(payload.get_subsystem_available_methods())
    print(payload.get_subsystem_available_settings())

    langmuir_reading = payload.read_value('LangmuirPayload')
    print(langmuir_reading.value)

    image_path = payload.read_value('Picture')
    print(image_path.value)

    orb.stop()
    rot.stop()


if __name__ == "__main__":
    # test_TestSensor()

    print("Starting TemperatureSensor test")
    test_TemperatureSensor()

    # print("Starting SunSensor test")
    # test_SunSensor()
    #
    # print("Starting EveryFaceSunSensor test")
    # test_EveryFaceSunSensor()
    #
    # print("Starting EveryFaceSunSensor with rotation test")
    # test_EveryFaceSunSensor_with_rotation()
    #
    # print("Starting GPSSensor test")
    # test_GPSSensor()
    #
    # print("Starting MagneticSimulation test")
    # test_MagneticSimulation()
    #
    # print("Starting MagnetometerSensor test")
    # test_MagnetometerSensor()
    #
    # print("Starting GyroscopeSensor test")
    # test_GyroscopeSensor()
    #
    # print("Starting CurrentSensor test")
    # test_CurrentSensor()
    #
    # print("Starting VoltageSensor test")
    # test_VoltageSensor()
    # 
    # print("Starting AccelerometerSensor test")
    # test_AccelerometerSensor()
    #
    # print("Starting LangmuirPayload Test")
    # test_LangmuirPayload()

    # print("Starting CameraPayload Test")
    # test_CameraPayload()
    #
    # print("Starting PayloadState Test")
    # test_PayloadState()

    # print("Starting COMMState test")
    # test_COMMState()

    # print("Starting ADCSState test")
    # test_ADCSState()

    # print("Starting EPSState test")
    # test_EPSState()

    # print("Starting SuchaiState test")
    # test_SuchaiState()

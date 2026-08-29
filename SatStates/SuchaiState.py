import random
import struct
import json
from typing import Union

import numpy as np
from PIL import Image
from utils.logger import Logger
import time
from Interfaces.Interface import Interface
from SatStates.SatState import SatState
from Sensors.SensorData import SensorData
from SatellitePersonality import SatellitePersonality


# from SubsystemsStates.RGBCameraState import RGBCameraState


class SuchaiState(SatState):

    def __init__(self, subsystem_states, interface: Interface):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        super().__init__(subsystem_states.values(), interface)
        self.subsystem_states = subsystem_states
        self.old_is_reachable = None

    def update(self, dt: Union[int, float]):
        # Process sat requests
        request = self.interface.request(timeout_ms=dt / 2)
        if request is not None:
            reply = self.process_request(request)
            self.interface.reply(reply)

            # logging request and reply
            _data_ = {
                'request': str(request),
                'reply': str(reply),
                'context': 'interface_request_and_replies'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        # TODO: Update states
        # Let's define the ZMQHub interface as a sensor/actuator, so we read if the satellite is reachable
        # and then enable/disable the ZMQHUB accordingly.
        # TODO Log all sensor data requests, is_reachable
        is_reachable = self.subsystem_states["comm"].read_value('is_comm_reachable').value
        self.subsystem_states["comm"].receive_command('set_comm_reachable', is_reachable)
        self.subsystem_states["adcs"].receive_command('update_dynamics', None)

        if self.old_is_reachable != is_reachable:
            self.old_is_reachable = is_reachable
            _data_ = {
                'is_reachable': is_reachable,
                'request': str(request),
                'context': 'comm_reachable'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(request),
                _data=_data_,
                _log_type='info'
            )

    def receive_command(self, command: str):
        pass

    def read_value(self, command: str) -> SensorData:
        pass

    def process_request(self, message: bytes) -> bytes:
        if message[0] == SatellitePersonality.SIM_OBC_ID:
            reply = self.process_obc_request(message)
        elif message[0] == SatellitePersonality.SIM_EPS_ID:
            reply = self.process_eps_request(message)
        elif message[0] == SatellitePersonality.SIM_ADCS_ID:
            reply = self.process_adcs_request(message)
        elif message[0] == SatellitePersonality.SIM_CAMERA_ID:
            reply = self.process_camera_request(message)
        else:
            reply = message
        return reply

    def process_obc_request(self, message: bytes) -> bytes:
        if message[1] == SatellitePersonality.SIM_OBC_ADDR_TEMP:
            temp = int(self.subsystem_states['eps'].read_value('temperature').value)
            reply = struct.pack('i', temp)
            print(f"OBC temp = {temp} ºC")

            _data_ = {
                'temp': temp,
                'reply': str(reply),
                'context': 'obs_temp'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        else:
            reply = message
        return reply

    def process_eps_request(self, message: bytes) -> bytes:
        if message[1] == SatellitePersonality.SIM_EPS_ADDR_HKP:
            vbat = int(self.subsystem_states['eps'].read_value('voltage').value)
            current_in = int(self.subsystem_states['eps'].read_value('current_in').value * 1000)
            current_out = int(self.subsystem_states['eps'].read_value('current_out').value * 1000)
            temp = int(self.subsystem_states['eps'].read_value('temperature').value - 273.15)
            # current_in = random.randint(0, 6000)  # mA
            # current_out = random.randint(0, 12000)  # mA
            # temp = random.randint(-40, 125)  # ºC
            reply = struct.pack('iiii', vbat, current_in, current_out, temp)
            print(f"EPS HK: VBat={vbat}, CurrIn={current_in}, CurrOut={current_out}, Temp={temp}")

            # logging
            _data_ = {
                'temp': temp,
                'vbat': vbat,
                'current_in': current_in,
                'current_out': current_in,
                'reply': str(reply),
                'context': 'eps_hk'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        elif message[1] == SatellitePersonality.SIM_EPS_ADDR_HEATER:
            mode = struct.unpack('B', message[2:])
            mode = mode[0] > 0  # True if mode > 0
            status = self.subsystem_states['eps'].receive_command('set_heater', mode)
            reply = struct.pack('i', status.value)

            # logging
            _data_ = {
                'mode': mode,
                'reply': str(reply),
                'context': 'eps_set_heater'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )
        elif message[1] == SatellitePersonality.SIM_EPS_ADDR_RESET:
            status = self.subsystem_states['eps'].receive_command('hard_reset', None)
            reply = struct.pack('i', status.value)

            # logging
            _data_ = {
                'status': status.value,
                'reply': str(reply),
                'context': 'eps_hard_reset'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )
        else:
            reply = message

        return reply

    def process_adcs_request(self, message: bytes) -> bytes:
        if message[1] == SatellitePersonality.SIM_ADCS_ADDR_MAG:
            x, y, z = self.subsystem_states['adcs'].read_value('magnetic_field_mGauss').value
            reply = struct.pack('fff', x, y, z)
            print(f"Mag: {x:.02f} {y:.02f} {z:.02f} mG")

            _data_ = {
                'x': x,
                'y': y,
                'z': z,
                'reply': str(reply),
                'context': 'magnetic_field_mGauss'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        elif message[1] == SatellitePersonality.SIM_ADCS_ADDR_GYR:
            all_data = self.subsystem_states['adcs'].read_value('gyroscope').value
            x, y, z = all_data['angular_velocity']
            reply = struct.pack('fff', x, y, z)
            print(f"Mag: {x:.02f} {y:.02f} {z:.02f} mG")

            _data_ = {
                'x': x,
                'y': y,
                'z': z,
                'reply': str(reply),
                'context': 'gyroscope'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        elif message[1] == SatellitePersonality.SIM_ADCS_ADDR_SUN:
            sun = self.subsystem_states['adcs'].read_value('in_sunlight').value
            reply = struct.pack('i', sun)
            print(f"Sun[{message[2]}] = {sun}")

            _data_ = {
                'sun': sun,
                'reply': str(reply),
                'context': 'adcs_sun'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        elif message[1] == SatellitePersonality.SIM_ADCS_ADDR_MTT:
            channel, duty = struct.unpack('bb', message[2:])
            try:
                assert (-100 < duty < 100), f"Torque duty cycle {duty} outside range"
                # Transform duty to torque 100% Duty -> MAX_TORQUE; -100% Duty => -MAX_TORQUE
                torque = SatellitePersonality.MAX_TORQUE_REACTION_WHEEL * duty / 100
                status = self.subsystem_states['adcs'].receive_command('set_torque_channel', (channel, torque))
            except ValueError:
                torque = 0;
                status = SensorData("status", False, bool)
            reply = struct.pack('i', status.value)
            # Log request
            _data_ = {
                'channel': channel,
                'duty': duty,
                'torque': torque,
                'status': status.value,
                'reply': str(reply),
                'context': 'set_torque_channel'
            }

            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )
        else:
            reply = message
        return reply

    def process_camera_request(self, message: bytes) -> bytes:
        if message[1] == SatellitePersonality.SIM_CAM_ADDR_SIZE:
            cam_size_x, cam_size_y = struct.unpack("ii", message[2:])
            status = self.subsystem_states["payload"].receive_command('set_size', (cam_size_x, cam_size_y))
            reply = struct.pack("i", status.value)

            _data_ = {
                'cam_size_x': cam_size_x,
                'cam_size_y': cam_size_y,
                'reply': str(reply),
                'context': 'camera_set_size'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )
        elif message[1] == SatellitePersonality.SIM_CAM_ADDR_TAKE:
            # img = Image.new("RGB", (SatellitePersonality.CAM_SIZE_X, SatellitePersonality.CAM_SIZE_Y))
            # img_path = "/tmp/img_{}.png".format(int(time.time()))
            # img.save(img_path)
            # TODO Log all sensor data requests, img_path
            img_path = self.subsystem_states['payload'].read_value('picture').value
            img_path = img_path.encode('ascii')
            img_path += b"\0" * (SatellitePersonality.SIM_CAM_PATH_LEN - len(img_path))
            print(f"Image Path = {img_path}")
            reply = img_path

            _data_ = {
                'img_path': img_path,
                'reply': str(reply),
                'context': 'camera_take_picture'
            }
            Logger.write_data(
                _class_name=SuchaiState.__class__.__name__,
                _id=id(reply),
                _data=_data_,
                _log_type='info'
            )

        else:
            reply = message

        return reply

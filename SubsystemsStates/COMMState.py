import zmq
import traceback
from multiprocessing import Process
from typing import Callable, Any

from Sensors.SensorData import SensorData
from Sensors.Sensor import Sensor
from SubsystemsStates.SubsystemState import SubsystemState
from Simulations.OrbitalSimulation import OrbitalSimulation


class COMMState(SubsystemState):

    def __init__(self, sensors: list[Sensor], in_port="8002", out_port="8001", mon_port="8003"):
        """
        Initialize the subsystem state. This method should be overridden by all subclasses.
        """
        super().__init__()

        # COMMState attributes
        self._subsystem_name = 'COMM'
        self._sensors_list = sensors

        # ZMQ HUB attributes
        self._in_port = in_port
        self._out_port = out_port
        self._mon_port = mon_port
        self._radio_on = False

        # Start zmq hub process and control socket
        self._ctx = zmq.Context()
        self._ctrl_port = "5557"
        self._s_ctrl = self._ctx.socket(zmq.PUSH)
        self._s_ctrl.bind("tcp://*:{}".format(self._ctrl_port))
        self._hub_process = Process(target=self.zmq_hub)
        self._hub_process.start()

        # SETTINGS
        self.ground_station_comm_reachability = False

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
        self.list_of_settable_methods.append(('set_comm_reachable', self.set_ground_station_comm_reachability))

        available_settings = self.list_of_settable_methods
        for single_data in available_settings:
            self.available_settings[single_data[0]] = single_data[1]

        # Disable ZMQ Hub at startup
        print(self._subsystem_name, "Radio mode off")
        self._radio_on = False
        self._s_ctrl.send(b"PAUSE")

    def get_subsystem_available_methods(self):
        return self.available_sensor_data

    def set_ground_station_comm_reachability(self, reachability):
        """
        Set ground_station_comm_reachability to either true or false.
        If true, the ZMQHub should continue communication with the FS.
        If false, the ZMQHub should be stopped and block communication between the Ground Station and the FS.
        """
        self.ground_station_comm_reachability = reachability
        self.set_radio_mode(self.ground_station_comm_reachability)

    def receive_command(self, command: str, value):
        """
        Method to receive a command.
        This needs to be implemented in the derived classes.
        """
        super().receive_command(command, value)

    def set_radio_mode(self, radio_on: bool):
        """
        Control the ZMQHub subprocess. If radio_on changes from False to True then start the ZMQHub subprocess,
        thus enabling communication. If radio_on changes from True to False, terminate the ZMQHub subprocess and
        disable communication. This method can be called multiple times, the ZMQHub is started/stopped only in
        case of radio_on value transition.

        :param radio_on: Enable or disable communications
        """
        if radio_on:
            if not self._radio_on:
                print(self._subsystem_name, "Radio mode on")
                self._radio_on = True
                self._s_ctrl.send(b"RESUME")
        else:
            if self._radio_on:
                print(self._subsystem_name, "Radio mode off")
                self._radio_on = False
                self._s_ctrl.send(b"PAUSE")

    def zmq_hub(self):
        """
        The ZMQHub subprocess. This enables communication between CSP nodes using the ZMQ.
        """
        # Create sockets
        context = zmq.Context()
        xpub_out = context.socket(zmq.XPUB)
        xsub_in = context.socket(zmq.XSUB)
        xpub_out.bind('{}://{}:{}'.format("tcp", "*", self._out_port))
        xsub_in.bind('{}://{}:{}'.format("tcp", "*", self._in_port))
        # Crate monitor socket
        s_mon = context.socket(zmq.PUB)
        s_mon.bind('tcp://*:{}'.format(self._mon_port))
        # Create control socket
        s_ctrl = context.socket(zmq.PULL)
        s_ctrl.connect('tcp://localhost:{}'.format(self._ctrl_port))

        # Start ZMQ proxy (blocking)
        try:
            zmq.proxy_steerable(xsub_in, xpub_out, s_mon, s_ctrl)
        except Exception as e:
            traceback.print_exc()
        finally:
            # print("Closing due to", e)
            xsub_in.setsockopt(zmq.LINGER, 0)
            xsub_in.close()
            xpub_out.close()
            s_ctrl.close()
            if s_mon:
                s_mon.close()

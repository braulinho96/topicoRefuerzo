import queue

from Simulations.Simulation import Simulation
from SatellitePersonality import SatellitePersonality
import numpy as np
from datetime import datetime
from threading import Lock
import matplotlib.pyplot as plt


class RotationSimulation(Simulation):

    def __init__(self, debug=False):
        super().__init__()
        self.moment_of_inertia = np.diag(np.array(SatellitePersonality.MOMENT_OF_INERTIA))
        self.inv_moment_of_inertia = np.linalg.inv(self.moment_of_inertia)
        self.max_torque_reaction_wheel = SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
        self.orientation = np.array([0.0, 0.0, 0.0])  # Euler angle 3-2-1 Yaw - Pitch - Roll
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # (i, j, k, s) Frame rotation from inertial vector to body frame
        self.angular_velocity = np.array([0.0, 5.0, -0.0]) * np.deg2rad(1)  # relative angular velocity of the spacecraft respect to inertial frame represented in body frame
        self.control_acceleration = np.array([0.0, 0.0, 0.0])
        self.torque = np.array([0.0, 0.0, 0.0])
        self.last_run_time = datetime.now()
        self._information_lock = Lock()

        # Setup debug plot
        self.__debug = debug  # Is debug enable
        self.__quat_hist = []  # Historic quaternion values
        self.__vel_hist = []  # Historic angular velocity values
        self.__time_hist = []  # Historic time
        if debug:
            self.__figure, self.__axes = plt.subplots(2, 1)
            self.__axes[0].grid(True)
            self.__axes[1].grid(True)
            plt.ion()
            plt.show(block=False)

    def _run_simulation(self):
        try:
            request = self._request_queue.get(timeout=0.1)
            if request is None:
                return

            with self._information_lock:
                self.update_simulation()

                if request.data == 'orientation':
                    result = tuple(self.orientation)
                elif request.data == 'quaternion':
                    result = tuple(self.quaternion)
                elif request.data == 'angular_velocity':
                    result = tuple(self.angular_velocity)
                elif request.data == 'angular_acceleration':
                    result = tuple(self.control_acceleration)
                elif request.data == 'torque':
                    result = tuple(self.torque)
                elif request.data == 'all':
                    result = {
                        'time': self.last_run_time.strftime('%Y-%m-%d_%H:%M:%S.%f'),
                        # 'time': self.last_run_time.strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3],
                        'orientation': tuple(self.orientation),
                        'angular_velocity': tuple(self.angular_velocity),
                        'angular_acceleration': tuple(self.control_acceleration),
                        'torque': tuple(self.torque)
                    }
                else:
                    result = 'Invalid request'

                self._set_result(request.id, result)
        except queue.Empty:
            # No more requests in the queue
            pass

        except Exception as e:
            # If an exception occurs, set it on the future
            if request and request.id in self._futures:
                self._set_exception(request.id, e)
            else:
                print(f"Exception in OrbitalSimulation: {e}")

    def update_simulation(self):
        current_time = datetime.now()
        time_difference = current_time - self.last_run_time
        time_difference_seconds = time_difference.total_seconds()

        if np.any(self.torque != 0):
            self.control_acceleration = self.inv_moment_of_inertia @ self.torque
        else:
            self.control_acceleration = np.array([0.0, 0.0, 0.0])

        # rk4 integration for time diff lower than 1 sec
        current_ang_velocity = self.angular_velocity.copy()
        self.angular_velocity += self.runge_kutta_4(self.d_omega, current_ang_velocity, time_difference_seconds)
        self.quaternion += self.runge_kutta_4(self.d_quaternion, self.quaternion, time_difference_seconds,
                                              current_ang_velocity)

        # normalization
        self.quaternion /= np.linalg.norm(self.quaternion)
        temp_orientation = self.q_to_ypr()
        # Only keep the orientation within the range of 0 to 2pi
        # self.orientation = np.mod(temp_orientation, 2 * np.pi)
        self.orientation = temp_orientation

        if self.__debug:
            print(self.quaternion, self.angular_velocity, time_difference_seconds)
            #self.__debug_plot(current_time, self.quaternion.tolist(), self.angular_velocity.tolist())

        self.last_run_time = current_time

    def __debug_plot(self, current_time, current_quaternion, current_velocity):
        """
        Update plot for debug purposes
        """
        self.__quat_hist.append(current_quaternion)
        self.__vel_hist.append(current_velocity)
        self.__time_hist.append(current_time)
        self.__axes[0].clear()
        self.__axes[0].plot(self.__time_hist, np.array(self.__vel_hist), "--.", label=["x", "y", "z"])
        self.__axes[0].legend(loc="upper right")
        self.__axes[0].set_ylabel('Velocity')
        self.__axes[0].set_xlabel('Time')
        self.__axes[0].grid(True)

        self.__axes[1].clear()
        self.__axes[1].plot(self.__time_hist, np.array(self.__quat_hist), "--.", label=["i", "j", "k", "s"])
        self.__axes[1].legend(loc="upper right")
        self.__axes[1].set_ylabel('Quaternion')
        self.__axes[1].set_xlabel('Time')
        self.__axes[1].grid(True)

        plt.title("Rotation Simulation")
        # plt.grid()
        plt.draw()
        plt.pause(0.001)

    def set_torque(self, torque: np.array):
        """
        Set torque vector
        :param torque: 3 axis torque values
        :return: None
        """
        if isinstance(torque, np.ndarray) and torque.shape == (3,):
            with self._information_lock:
                self.update_simulation()

                if np.any(np.abs(torque) > self.max_torque_reaction_wheel):
                    raise ValueError('Torque exceeds the maximum torque of the reaction wheel')

                self.torque = torque
        else:
            raise ValueError('Torque must be a numpy array with shape (3,)')

    def get_torque(self) -> np.array:
        """
        Return current torque values
        :return: 3 axis torque values
        """
        return self.torque

    def q_to_ypr(self):
        dcm = self.to_dcm()
        yaw = np.arctan2(dcm[0, 1], dcm[0, 0])
        if abs(dcm[0, 2]) > 1:
            dcm[0, 2] = abs(dcm[0, 2]) / dcm[0, 2]
        pitch = np.arcsin(-dcm[0, 2])
        roll = np.arctan2(dcm[1, 2], dcm[2, 2])
        return yaw, pitch, roll

    def to_dcm(self):
        q1 = self.quaternion[0]
        q2 = self.quaternion[1]
        q3 = self.quaternion[2]
        q4 = self.quaternion[3]

        dcm = [[q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q1 * q2 + q3 * q4), 2 * (q1 * q3 - q2 * q4)],
               [2 * (q1 * q2 - q3 * q4), -q1 ** 2 + q2 ** 2 - q3 ** 2 + q4 ** 2, 2 * (q2 * q3 + q1 * q4)],
               [2 * (q1 * q3 + q2 * q4), 2 * (q2 * q3 - q1 * q4), -q1 ** 2 - q2 ** 2 + q3 ** 2 + q4 ** 2]]
        return np.array(dcm)

    def d_quaternion(self, x_quaternion_i2b, x_omega_b):
        ok = self.omega4kinematics(x_omega_b)
        q_dot = 0.5 * ok @ x_quaternion_i2b
        return q_dot

    def d_omega(self, x_omega_b: np.array) -> np.array:
        h_total_b = self.moment_of_inertia.dot(x_omega_b)
        w_dot = - self.inv_moment_of_inertia @ (np.cross(x_omega_b, h_total_b))
        return w_dot

    @staticmethod
    def omega4kinematics(x_omega_b: np.array):
        omega_4x4 = np.zeros((4, 4))
        omega_4x4[1, 0] = -x_omega_b[2]
        omega_4x4[2, 0] = x_omega_b[1]
        omega_4x4[3, 0] = -x_omega_b[0]

        omega_4x4[0, 1] = x_omega_b[2]
        omega_4x4[0, 2] = -x_omega_b[1]
        omega_4x4[0, 3] = x_omega_b[0]

        omega_4x4[1, 2] = x_omega_b[0]
        omega_4x4[1, 3] = x_omega_b[1]

        omega_4x4[2, 1] = -x_omega_b[0]
        omega_4x4[2, 3] = x_omega_b[2]

        omega_4x4[3, 1] = -x_omega_b[1]
        omega_4x4[3, 2] = -x_omega_b[2]
        return omega_4x4

    @staticmethod
    def quat_mut(left_quat, right_quat):
        temp = np.zeros(4)
        temp[0] = (left_quat[3] * right_quat[0] - left_quat[2] * right_quat[1] + left_quat[1] * right_quat[2] +
                   left_quat[0] * right_quat[3])
        temp[1] = (left_quat[2] * right_quat[0] + left_quat[3] * right_quat[1] - left_quat[0] * right_quat[2] +
                   left_quat[1] * right_quat[3])
        temp[2] = (-left_quat[1] * right_quat[0] + left_quat[0] * right_quat[1] + left_quat[3] * right_quat[2] +
                   left_quat[2] * right_quat[3])
        temp[3] = (-left_quat[0] * right_quat[0] - left_quat[1] * right_quat[1] - left_quat[2] * right_quat[2] +
                   left_quat[3] * right_quat[3])
        return temp

    @staticmethod
    def runge_kutta_4(function, x, dt, *args):
        k1 = function(x, *args)
        xk2 = x + (dt / 2.0) * k1

        k2 = function(xk2, *args)
        xk3 = x + (dt / 2.0) * k2

        k3 = function(xk3, *args)
        xk4 = x + dt * k3

        k4 = function(xk4, *args)

        next_x = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_x
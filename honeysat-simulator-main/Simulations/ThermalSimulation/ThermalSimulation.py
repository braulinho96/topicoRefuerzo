import queue

from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.Simulation import Simulation
import time
from scipy.constants import Stefan_Boltzmann


class ThermalSimulation(Simulation):
    __specific_heat: float  # J/(kg*K)
    __mass: float  # kg
    __temperature: float  # K
    __last_run_time: float  # s
    __emissivity: float  # 0-1 (0: perfect reflector, 1: perfect black body)
    __surface_area: float  # m^2

    __osim: OrbitalSimulation

    # The values in the constructor are specific to the satellite material components.
    # For example, material like aluminium which is represented in the specific_heat
    def __init__(self, osim: OrbitalSimulation, specific_heat: float = 903, mass: float = 1.0,
                 temperature: float = 30 + 273, emissivity: float = 0.7, surface_area: float = 0.03):
        super().__init__()
        self.__osim = osim
        self.__specific_heat = specific_heat
        self.__mass = mass
        self.__temperature = temperature
        self.__last_run_time = time.time()
        self.__emissivity = emissivity
        self.__surface_area = surface_area

    def _run_simulation(self):
        try:
            request = self._request_queue.get(timeout=0.1)
            if request is None:
                return

            match request.data:
                case "get_temperature":
                    temp = self.get_temperature()
                    result = {
                        'temperature': temp,
                    }
                case _:
                    result = {

                    }

            # Set the result for the corresponding future
            self._set_result(request.id, result)

        except queue.Empty:
            # No more requests in the queue
            self.update_simulation()

        except Exception as e:
            # If an exception occurs, set it on the future
            if request and request.id in self._futures:
                self._set_exception(request.id, e)
            else:
                print(f"Exception in ThermalSimulation: {e}")

    def update_simulation(self):
        current_time = time.time()
        delta_time = current_time - self.__last_run_time
        self.__last_run_time = current_time

        # radiation loss
        delta_p = -1 * Stefan_Boltzmann * self.__emissivity * self.__surface_area * self.__temperature ** 4

        # sun
        if self.__osim.send_request('is_sun_present').result()['is_sunlit']:
            delta_p += 1367.0 * self.__surface_area * self.__emissivity

        delta_e = delta_p * delta_time

        delta_t = delta_e / (self.__mass * self.__specific_heat)
        self.__temperature += delta_t

    def get_temperature(self):
        return self.__temperature

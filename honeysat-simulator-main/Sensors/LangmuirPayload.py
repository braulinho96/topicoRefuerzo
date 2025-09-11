from typing import Callable
from numpy import interp, arange
from datetime import datetime
from random import randint

from Sensors.Sensor import Sensor
from Sensors.Sensor import SensorData

from Simulations.OrbitalSimulation import OrbitalSimulation


class LangmuirPayload(Sensor):
    
    def __init__(self, orbital_simulation: OrbitalSimulation, noise=5):
        super().__init__()
        self._sensor_name = "LangmuirPayload"

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be a OrbitalSimulation")

        self.orbital_simulation = orbital_simulation

        self._noise = noise

        self._lp_index = 0
        self._hk_idx = 0
        self._lp_hgchn = [452, 470, 471, 482, 487, 492, 498, 511, 514, 522, 529, 531, 532, 533, 534, 537,
                          538, 539, 543, 544, 545, 548, 553, 554, 559, 561, 563, 579, 583, 605]

        self._lp_lgchn = [309, 310, 311, 312, 313, 314, 315, 316, 317]

        self._lp_hk = [0, 70, 77, 79, 902, 905, 906, 907, 908, 909, 911, 936, 937, 940, 941, 983, 984, 985, 986]
    
    def get_value(self):

        result = self.orbital_simulation.send_request('get_azimuth').result()
        azimuth = result['satellite_azimuth'].degrees

        lp_hgchn = interp(azimuth, arange(start=0, stop=360, step=(360/len(self._lp_hgchn))), self._lp_hgchn)
        lp_lgchn = interp(azimuth, arange(start=0, stop=360, step=(360 / len(self._lp_lgchn))), self._lp_lgchn)
        lp_hk = interp(azimuth, arange(start=0, stop=360, step=(360 / len(self._lp_hk))), self._lp_hk)

        reading ={
            'lp_index': self._lp_index,
            'lp_timestamp': int(datetime.now().timestamp()),
            'lp_unit': 6,
            'hk_idx': self._hk_idx,
            'lp_hk': int(lp_hk) + randint((-1 * self._noise), self._noise),
            'lp_hgchn': int(lp_hgchn) + randint((-1 * self._noise), self._noise),
            'lp_lgchn': int(lp_lgchn) + randint((-1 * self._noise), self._noise),
            'crc': randint(200, 100000),
            'chk': 1
        }

        self._lp_index += 1
        self._hk_idx = (self._hk_idx + 1) % 6

        return SensorData(name='reading', value=reading, data_type=type(reading))

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:
        return [('LangmuirPayload', self.get_value)]

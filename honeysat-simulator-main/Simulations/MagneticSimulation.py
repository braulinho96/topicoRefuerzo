import queue

from Simulations.Simulation import Simulation
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.RotationSimulation import RotationSimulation

from SatellitePersonality import SatellitePersonality

import numpy as np
from datetime import datetime
import pyIGRF


class MagneticSimulation(Simulation):

    def __init__(self, orbital_simulation: OrbitalSimulation, rotation_simulation: RotationSimulation):
        super().__init__()

        if not OrbitalSimulation or not isinstance(orbital_simulation, OrbitalSimulation):
            raise ValueError('orbital_simulation must be an instance of OrbitalSimulation')
        self.orbital_simulation = orbital_simulation

        if not RotationSimulation or not isinstance(rotation_simulation, RotationSimulation):
            raise ValueError('rotation_simulation must be an instance of RotationSimulation')
        self.rotation_simulation = rotation_simulation

        self.satellite_magnetic_field = np.array([0.0, 0.0, 0.0])
        self.generated_torque = np.array([0.0, 0.0, 0.0])
        self.last_run_time = datetime.now()

    def _run_simulation(self):
        if self._check_time_elapsed() > 1:
            self.update_simulation()

        try:
            request = self._request_queue.get(timeout=0.1)
            if request is None:
                return

            if request.data == 'internal_satellite_magnetic_field':
                # TODO NOT IMPLEMENTED YET
                result = tuple(self.satellite_magnetic_field)
            if request.data == 'earth_magnetic_field':
                magnetic_field = self.get_earth_magnetic_field()
                """
                OUTPUT
                     x     = north component (nT) if isv = 0, nT/year if isv = 1
                     y     = east component (nT) if isv = 0, nT/year if isv = 1
                     z     = vertical component (nT) if isv = 0, nT/year if isv = 1
                     f     = total intensity (nT) if isv = 0, rubbish if isv = 1
                 """
                result = {
                    'north': magnetic_field[0],
                    'east': magnetic_field[1],
                    'vertical': magnetic_field[2],
                    'total_intensity': magnetic_field[3],
                    'time': datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3],
                }
            elif request.data == 'all':
                result = {
                    'time': self.last_run_time.strftime('%Y-%m-%d_%H:%M:%S.%f'),
                    # 'time': self.last_run_time.strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3],
                    'satellite_magnetic_field': tuple(self.satellite_magnetic_field),
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
        # TODO NOT IMPLEMENTED YET
        current_time = datetime.now()

    def _check_time_elapsed(self):
        current_time = datetime.now()
        time_difference = current_time - self.last_run_time
        time_difference_seconds = time_difference.total_seconds()
        return time_difference_seconds

    def get_earth_magnetic_field(self):
        """
        OUTPUT
             x     = north component (nT) if isv = 0, nT/year if isv = 1
             y     = east component (nT) if isv = 0, nT/year if isv = 1
             z     = vertical component (nT) if isv = 0, nT/year if isv = 1
             f     = total intensity (nT) if isv = 0, rubbish if isv = 1
         """
        gps_data = self.orbital_simulation.send_request('get_gps_data').result()

        lat = gps_data['satellite_latitude'].degrees
        lon = gps_data['satellite_longitude'].degrees
        alt = gps_data['satellite_altitude_surface_km']

        # Get the magnetic field of the earth at the current position
        magnetic_field = pyIGRF.calculate.igrf12syn(date=datetime.now().year, itype=1, alt=alt, lat=lat, elong=lon)
        return magnetic_field

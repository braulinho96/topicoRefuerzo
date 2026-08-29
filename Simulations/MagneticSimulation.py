import queue

from Simulations.Simulation import Simulation
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.RotationSimulation import RotationSimulation

from SatellitePersonality import SatellitePersonality

import numpy as np
from datetime import datetime
import pyIGRF
from skyfield.api import utc


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
        self.magnetic_field = None
        self.generated_torque = np.array([0.0, 0.0, 0.0])
        self.last_run_time = datetime.now()

    def _run_simulation(self):
        if self._check_time_elapsed() > 1:
            self.update_simulation(datetime.now())

        try:
            request = self._request_queue.get(timeout=0.1)
            if request is None:
                return

            if request.data == 'internal_satellite_magnetic_field':
                # TODO NOT IMPLEMENTED YET
                result = tuple(self.satellite_magnetic_field)
            if request.data == 'earth_magnetic_field':
                magnetic_field = self.get_earth_magnetic_field(datetime.now())
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

    def update_simulation(self, current_time: datetime):
        self.last_run_time = current_time
        self.magnetic_field = self.get_earth_magnetic_field(current_time)
    

    def get_magnetic_field(self) -> np.ndarray:
        """
        Retorna el campo magnético de la Tierra en el marco de coordenadas Inercial (nT).
        """
        return self.magnetic_field

    def _check_time_elapsed(self):
        current_time = datetime.now()
        time_difference = current_time - self.last_run_time
        time_difference_seconds = time_difference.total_seconds()
        return time_difference_seconds

    def get_earth_magnetic_field(self, current_time: datetime):
        """
        OUTPUT
             x     = north component (nT) if isv = 0, nT/year if isv = 1
             y     = east component (nT) if isv = 0, nT/year if isv = 1
             z     = vertical component (nT) if isv = 0, nT/year if isv = 1
             f     = total intensity (nT) if isv = 0, rubbish if isv = 1
         """
        # Directly use the attributes from the orbital simulation object
        lat = self.orbital_simulation.get_latitude().degrees
        lon = self.orbital_simulation.get_longitude().degrees
        alt = self.orbital_simulation.get_alt_from_surface_km()

        # Get the magnetic field of the earth at the current position
        # pyIGRF can use a decimal year, which is more precise
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=utc)
            
        time_fraction = (current_time - datetime(current_time.year, 1, 1, tzinfo=utc)).total_seconds() / \
                        (datetime(current_time.year + 1, 1, 1, tzinfo=utc) - datetime(current_time.year, 1, 1, tzinfo=utc)).total_seconds()
        decimal_year = current_time.year + time_fraction

        magnetic_field = pyIGRF.calculate.igrf12syn(date=decimal_year, itype=1, alt=alt, lat=lat, elong=lon)
        return magnetic_field

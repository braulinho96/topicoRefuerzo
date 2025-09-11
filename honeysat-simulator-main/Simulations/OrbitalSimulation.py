import os
from datetime import datetime, timedelta
from queue import Empty

from scipy.spatial.transform import Rotation as R
import numpy as np

from Simulations.Simulation import Simulation
from Simulations.RotationSimulation import RotationSimulation

from SatellitePersonality import SatellitePersonality
from skyfield.api import Loader, wgs84, Topos

# Create skyfield loader with custom path
load = Loader('./TLE_and_data')


class OrbitalSimulation(Simulation):

    def __init__(self, rotation_simulation: RotationSimulation):
        super().__init__()

        # Store the rotation simulation
        if not RotationSimulation or not isinstance(rotation_simulation, RotationSimulation):
            raise ValueError('rotation_simulation must be an instance of RotationSimulation')
        self.rotation_simulation = rotation_simulation

        # TLE CREATION
        self.norad_catalog_number = SatellitePersonality.NORAD_CATALOG_NUMBER
        self.tle_filename = 'tle-CATNR-{}.txt'.format(self.norad_catalog_number)

        # Create a satellite from TLE data
        self.last_tle_update = None
        self.satellite = self.__download_satellite_tle()
        self.last_tle_update = datetime.now()

        # Download the ephemeris data
        self.eph = load('de421.bsp')

        # Create an observer at the specified latitude and longitude
        self.observer = Topos(latitude_degrees=SatellitePersonality.OBSERVER_LATITUDE,
                              longitude_degrees=SatellitePersonality.OBSERVER_LONGITUDE)

        self.last_orbital_update = None
        self.last_orientation_update = None

        self.__sun_orientation = np.array([0.0, 0.0, 0.0])
        self.__earth_orientation = np.array([0.0, 0.0, 0.0])

        # Initialize all attributes to default values
        self.__distance_to_earth_center = 0.0
        self.__latitude = 0.0
        self.__longitude = 0.0
        self.__alt_from_surface_km = 0.0
        self.__altitude_degrees = 0.0
        self.__azimuth = 0.0
        self.__distance = 0.0
        self.__sunlit = False
        self.__satellite_subpoint_location = None

    def _run_simulation(self):
        self.__update_full_simulation_data()
        try:
            request = self._request_queue.get(timeout=0.1)
            if request is None:
                return

            match request.data:
                case "get_lat_lon":
                    result = {
                        'satellite_latitude': self.__latitude,
                        'satellite_longitude': self.__longitude,
                    }
                case "get_gps_data":
                    result = {
                        'satellite_latitude': self.__latitude,
                        'satellite_longitude': self.__longitude,
                        'satellite_altitude_degrees': self.__altitude_degrees,
                        'satellite_altitude_surface_km': self.__alt_from_surface_km,
                    }

                case "get_comm_reachable_data":
                    result = {
                        'satellite_altitude_degrees': self.__altitude_degrees,
                        'satellite_altitude_distance': self.__distance,
                    }
                case "get_altitude":
                    result = {
                        'satellite_altitude_degrees': self.__altitude_degrees,
                    }
                case "get_azimuth":
                    result = {
                        'satellite_azimuth': self.__azimuth,
                    }
                case "is_sun_present":
                    result = {
                        'is_sunlit': self.__sunlit,
                    }
                case "get_subpoint_location":
                    result = {
                        'orbital_subpoint_location': self.__satellite_subpoint_location,
                    }
                case "get_sun_orientation":
                    result = {
                        'orbital_sun_orientation': self.__sun_orientation,
                    }
                case "get_earth_orientation":
                    result = {
                        'orbital_earth_orientation': self.__earth_orientation,
                    }
                case "get_distance_to_earth_center":
                    result = {
                        'distance_to_earth_center': self.__distance_to_earth_center,
                    }
                case "get_all_orbital":
                    result = {
                        'satellite_latitude': self.__latitude,
                        'satellite_longitude': self.__longitude,
                        'satellite_altitude_degrees': self.__altitude_degrees,
                        'satellite_azimuth': self.__azimuth,
                        'is_sunlit': self.__sunlit,
                        'satellite_subpoint_location': self.__satellite_subpoint_location,
                        'sun_orientation': self.__sun_orientation,
                    }

                case _:
                    result = {}

            # Set the result for the corresponding future
            self._set_result(request.id, result)

        except Empty:
            # No more requests in the queue
            pass

        except Exception as e:
            # If an exception occurs, set it on the future
            if request and request.id in self._futures:
                self._set_exception(request.id, e)
            else:
                print(f"Exception in simulation: {e}")

    def __update_tle_if_needed(self):
        """ Update the TLE file if it is too old. """
        if self.__is_tle_too_old():
            print("Downloading new updated TLE file - it older than {} days".format(SatellitePersonality.TLE_MAX_AGE))
            self.satellite = self.__download_satellite_tle()
            self.last_tle_update = datetime.now()

    def __download_satellite_tle(self):
        """ Download the most recent TLE file from
        the specified satellite from the online norad catalog.
        """
        norad_url = 'https://celestrak.org/NORAD/elements/gp.php?CATNR={}'.format(self.norad_catalog_number)
        satellite_list = load.tle_file(norad_url, filename=self.tle_filename, reload=self.__is_tle_too_old())
        # load.tle_file returns a list, but we only need the first satellite in the list.
        satellite = satellite_list[0]
        return satellite

    def __is_tle_too_old(self) -> bool:
        """
        Check whether the current TLE file age is older than TLE_MAX_AGE in hours.
        Returns True if the file is too old or does not exist.
        """

        if self.last_tle_update is not None:
            # Check if the file is too old
            return (datetime.now() - self.last_tle_update) > timedelta(hours=SatellitePersonality.TLE_MAX_AGE)
        else:
            try:
                # Check if the file is too old
                # multiply by 24 to convert days to hours
                return load.days_old(self.tle_filename) * 24 > SatellitePersonality.TLE_MAX_AGE
            except FileNotFoundError:
                # File doesn't exist, so it's considered 'too old'
                return True
            except OSError as e:
                # Handle other OS-level errors (like permission issues)
                print(f"Error accessing file: {e}")
                return True

    def __update_full_simulation_data(self):
        """
        Update all the simulation data if it is too old.
        """
        if self.__is_orientation_data_too_old():
            self.__update_orientation_data()
            self.last_orientation_update = datetime.now()

        if self.__is_orbital_data_too_old():
            self.__update_orbital_data()
            self.last_orbital_update = datetime.now()

        self.__update_tle_if_needed()

    def __is_orientation_data_too_old(self) -> bool:
        """
        Check whether the current orientation data is older than 1 second.
        Returns True if the data is too old or does not exist.
        """

        if self.last_orientation_update is not None:
            # Check if the file is too old
            return (datetime.now() - self.last_orientation_update) > timedelta(seconds=1)
        else:
            return True

    def __update_orientation_data(self):
        """
        Update the orientation data of the satellite.
        It makes calculations based on the current satellite position and the position of the sun and earth.
        It uses the rotation simulation to get the orientation of the satellite.
        """

        ts = load.timescale()
        now = ts.now()

        eph = self.eph
        earth = eph['earth']
        sun = eph['sun']

        # get the Rotation from the rotation simulation
        satellite_orientation = np.array(self.rotation_simulation.send_request('quaternion').result())
        satellite_orientation[:3] *= -1  # conjugate
        rotation_satellite = R.from_quat(satellite_orientation)

        # -----------------------------------------
        # SUN ORIENTATION
        sat_SSB_location = earth + self.satellite

        # Vector from the satellite to the sun ~ distance is plausible
        distance_sat_sun = (sun - sat_SSB_location).at(now).position.au

        norm = np.linalg.norm(distance_sat_sun)
        normalized_distance = distance_sat_sun / norm

        # Apply the rotation to the normalized vector to get the relative position of the sun
        self.__sun_orientation = rotation_satellite.apply(normalized_distance)

        # -----------------------------------------
        # EARTH ORIENTATION
        sat_to_earth = - self.satellite

        # Vector from the satellite to the earth ~ distance is plausible
        distance_sat_earth = sat_to_earth.at(now).position.km

        norm = np.linalg.norm(distance_sat_earth)
        normalized_distance = distance_sat_earth / norm

        # Apply the rotation to the normalized vector to get the relative position of the sun
        self.__earth_orientation = rotation_satellite.apply(normalized_distance)

    def __is_orbital_data_too_old(self) -> bool:
        """
        Check whether the current orbital data is older than 10 seconds.
        Returns True if the data is too old or does not exist.
        """

        if self.last_orbital_update is not None:
            # Check if the file is too old
            return (datetime.now() - self.last_orbital_update) > timedelta(seconds=10)
        else:
            return True

    def __update_orbital_data(self):
        """
        It updates the orbital data of the satellite.
        It makes calculations based on the current satellite position and the position of the observer.
        """
        ts = load.timescale()
        now = ts.now()

        # Compute the position of the satellite at the current time
        geocentric = self.satellite.at(now)

        # Compute the position of the satellite relative to observer (ground station)
        difference = self.satellite - self.observer
        topocentric = difference.at(now)

        # TODO CHECK CORRECTNESS
        # Old get_distance_to_earth_center()
        self.__distance_to_earth_center = geocentric.distance().km

        # Old get_orbital_lat_lon()
        self.__latitude, self.__longitude = wgs84.latlon_of(geocentric)

        # Old get_gps_data()
        self.__alt_from_surface_km = wgs84.height_of(geocentric).km

        self.__altitude_degrees, self.__azimuth, self.__distance = topocentric.altaz()

        # Old is_sunlit()
        self.__sunlit = self.satellite.at(now).is_sunlit(self.eph)

        # Old get_orbital_subpoint_location()
        self.__satellite_subpoint_location = wgs84.subpoint_of(geocentric)

import os
from random import randint
from typing import Callable
from PIL import Image

from Sensors.Sensor import Sensor
from Sensors.Sensor import SensorData

from Simulations.OrbitalSimulation import OrbitalSimulation


class CameraPayload(Sensor):

    def __init__(self, orbital_simulation: OrbitalSimulation):
        super().__init__()

        self._sensor_name = "CameraPayload"
        self._history = []
        self._max_size = 20
        self._width = 1024
        self._height = 1024

        self._day_img_repository = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/daytime')
        self._night_img_repository = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/nighttime')
        if not os.path.exists(self._day_img_repository):
            raise FileNotFoundError("No day image directory found")
        if not os.path.exists(self._night_img_repository):
            raise FileNotFoundError("No night image directory found")

        self._num_of_day_images = len(os.listdir(self._day_img_repository))
        self._num_of_night_images = len(os.listdir(self._night_img_repository))

        if not orbital_simulation:
            raise ValueError("No orbital_simulation available")
        if not isinstance(orbital_simulation, OrbitalSimulation):
            raise TypeError("Simulation must be a OrbitalSimulation")

        self._orbital_simulation = orbital_simulation

    def _find_image_path(self, directory: str, num_of_imgs: int) -> str:
        img_path = ''

        folder_1024 = os.path.join(directory, "{}x{}".format(1024, 1024))

        while True:
            img_path = os.listdir(folder_1024)[
                randint(0, num_of_imgs)
            ]

            img_path = os.path.basename(img_path)

            if img_path not in self._history:
                if len(self._history) >= self._max_size:
                    self._history.pop(0)

                self._history.append(img_path)
                break

        return img_path

    def get_image(self):

        result = self._orbital_simulation.send_request('is_sun_present').result()

        # TODO - Check Camera's Orientation w.r.t Earth

        directory = self._day_img_repository

        if not result['is_sunlit']:
            directory = self._night_img_repository

        img_pth = self._find_image_path(directory, self._num_of_day_images)
        img_pth = self._change_image_size(directory, img_pth)
        img_pth = os.path.join(directory, "{}x{}/{}".format(self._width, self._height, img_pth))

        return SensorData(name='path', value=img_pth)

    def _change_image_size(self, directory: str, path: str) -> str:
        folder = os.path.join(directory, "{}x{}".format(self._width, self._height))
        folder_1024 = os.path.join(directory, "1024x1024")

        if not os.path.exists(folder):
            os.makedirs(folder)

        if not os.path.exists(os.path.join(folder, path)):
            image = Image.open(os.path.join(folder_1024,path))
            new_image = image.resize((self._width, self._height))
            image.close()

            new_image.save(os.path.join(folder, path))
            new_image.close()

        return path

    def set_image_size(self, *args):
        status_code = 200
        size = args[0]

        try:
            self._width = int(size[0])
            self._height = int(size[1])
        except TypeError:
            status_code = 301
        except ValueError:
            status_code = 302
        except IndexError:
            status_code = 303
        except Exception as e:
            status_code = 555

        return SensorData(name='config', value=status_code)

    def _generate_sensor_list_of_methods(self) -> list[tuple[str, Callable[[], SensorData]]]:

        return [('picture', self.get_image)]

    def get_sensor_list_of_settings(self) -> list[tuple[str, Callable[[], SensorData]]]:
        # Add orientation stabilization mode?
        return [('set_size', self.set_image_size)]

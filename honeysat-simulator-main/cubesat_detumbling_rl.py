# Librerías a utilizar
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import matplotlib
from zmq.backend import second

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Importar componentes existentes del simulador de HoneySat
from Simulations.RotationSimulation import RotationSimulation
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.MagneticSimulation import MagneticSimulation
from SatellitePersonality import SatellitePersonality
from skyfield.api import utc


class CubeSatDetumblingEnv(gym.Env):
    """
    ENTORNO DE GYMNASIUM PARA PROBLEMA DE DETUMBLING USANDO SIMULADOR DE HONEYSAT.

    Este entorno se integra con las clases existentes de RotationSimulation, OrbitalSimulation
    y MagneticSimulation para proporcionar una simulación realista de la dinámica de satélites
    para el aprendizaje por refuerzo.
    """

    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, render_mode=None, max_steps=500, start_time=datetime.now(), time_step=0.1, granularity=100, debug=False, plot_hist=False):
        """
        Inicializar el entorno de CubeSat para el problema de detumbling.

        Args:
            render_mode (str): Modo de renderizado ('human' o None)
            max_steps (int): Pasos máximos por episodio
            start_time (datetime): Tiempo inicial de la simulacion
            time_step (float): Paso de tiempo de simulación en segundos
            granularity (int): Granularida de la simulacion, divide a time_step
            debug (bool): Activar historico de observaciones y graficar
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.time_step = time_step
        self.start_time = start_time
        self.current_time = self.start_time

        self.sim_granularity = granularity
        self._plot_hist = plot_hist

        # inicializar componentes del simulador
        self.rotation_sim = None
        self.orbital_sim = None
        self.magnetic_sim = None

        # Discretize the action space for Q-learning
        # Actions: Positive/Negative torque on each axis (X, Y, Z) + No torque
        self.max_torque = SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
        self.action_map = {
            0: np.array([self.max_torque, 0, 0]),
            1: np.array([-self.max_torque, 0, 0]),
            2: np.array([0, self.max_torque, 0]),
            3: np.array([0, -self.max_torque, 0]),
            4: np.array([0, 0, self.max_torque]),
            5: np.array([0, 0, -self.max_torque]),
            6: np.array([0, 0, 0]),  # No torque
        }
        self.action_space = spaces.Discrete(len(self.action_map))

        # definir espacio de observaciones
        # box: quaternion (4) + velocidad angular (3) + campo magnetico (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        # tracking dentro de un episodio
        self.current_step = 0
        self.episode_reward = 0.0

        # como es casi imposible tener velocidad angular cero, se establece un umbral
        # ajustable dependiendo de la misión y el contexto
        self.success_threshold = 0.01  # rad/s

        # Para efectos de debug, guardar un historial de observaciones y graficarlas
        self._debug = debug  # Debug activado
        self._observation_hist = []  # Historico de observaciones
        self._time_hist = []  # Historic time
        if self._debug:
            # import matplotlib.pyplot as plt
            pass
            # self.__figure, axes = plt.subplots(2, 1)
            # axes[0].grid(True)
            # axes[1].grid(True)
            # plt.ion()
            # plt.show(block=False)

    def _create_simulators(self):
        """Crear instancias de simuladores.
        - RotationSimulation
        - OrbitalSimulation
        - MagneticSimulation
        """
        # llamar constructores de simuladores, ver parametros si se necesita debuguear
        self.rotation_sim = RotationSimulation(debug=False, start_time=self.start_time)
        self.orbital_sim = OrbitalSimulation(self.rotation_sim)
        self.magnetic_sim = MagneticSimulation(self.orbital_sim, self.rotation_sim)


    def _start_simulators(self):
        """Inicializar hilos de cada simulador. Implementación paralelizada."""
        # This is no longer needed for the synchronous RL environment
        pass

    def _stop_simulators(self):
        """Detener los hilos de cada simulador."""
        # This is no longer needed for the synchronous RL environment
        pass

    def _update_simulators(self, current_time: datetime):
        """ Actualizar los simuladores segun el tiempo simulado """
        self.rotation_sim.update_simulation(current_time)
        self.orbital_sim.update_simulation(current_time)
        self.magnetic_sim.update_simulation(current_time)

    def reset(self, seed=None, options=None):
        """
        Función para reiniciar el entorno y comenzar un nuevo episodio.
        Args:
            seed (int): Semilla aleatoria para reproducibilidad
            options (dict): Opciones adicionales (no utilizadas)

        Returns:
            tuple: (observación, información)
        """
        super().reset(seed=seed)

        # parar simulaciones para luego reiniciarlas para nuevo episodio
        self._create_simulators()

        # condiciones iniciales aleatorias o fijas
        initial_angular_velocity = self.np_random.uniform(-1.0, 1.0, size=3)
        # initial_angular_velocity = np.array([1.0, 0.0, 0.0])

        initial_quat = self.np_random.normal(size=4)
        # initial_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Setear condiciones iniciales
        initial_quat /= np.linalg.norm(initial_quat)
        self.rotation_sim.angular_velocity = initial_angular_velocity
        self.rotation_sim.quaternion = initial_quat

        # empezar simulaciones con nuevas condiciones
        # self._start_simulators()

        # reiniciar tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_time = self.start_time

        self.orbital_sim.update_simulation(self.current_time)
        self.magnetic_sim.update_simulation(self.current_time)

        observation = self._get_observation()
        info = {}

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Ejecuta un solo paso en el entorno dentro de un episodio.

        Args:
            action (np.ndarray): Comando en 3 dimensiones representando el torque

        Returns:
            tuple: (observación, recompensa, terminado, truncado, información) /
                   (observation, reward, terminated, truncated, info)
        """
        # mapear accion discreta a vector de torque
        torque_action = self.action_map[action]

        ### TEST: Compare with a simple proportional controller
        # G = 1e-3
        # torque_action = -self.rotation_sim.angular_velocity*G
        ###

        # aplicar accion de torque al simulador de rotacion
        self.rotation_sim.set_torque(torque_action)

        # Avanzar la simulacion con una granularidad menor
        dt = self.time_step / self.sim_granularity
        for i in range(self.sim_granularity):
            self.current_time += timedelta(seconds=dt) # Avanzar el tiempo en el step definido
            self._update_simulators(self.current_time) # Actualizar las simulaciones


            # obtener nueva observacion
            observation = self._get_observation()
            # Guardar historicos para graficar
            if self._debug:
                # Agregar el torque también al historico
                observation = np.concatenate((observation, torque_action))
                self._observation_hist.append(observation)
                # Agregar el tiempo al historico
                self._time_hist.append(self.current_time.timestamp())

        # calcular recompensa
        reward = self._calculate_reward(torque_action)
        self.episode_reward += reward

        # revisar si es que termino el episodio
        try:
            angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)
            terminated = angular_vel_norm < self.success_threshold
        except Exception:
            angular_vel_norm = 1.0
            terminated = False

        # revisar si ocurre un timeout episodico
        # parametro "truncated" (revisar docs en gymnasium)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # retornar info adicional
        info = {
            'angular_velocity_norm': angular_vel_norm,
            'episode_reward': self.episode_reward,
            'success': terminated
        }

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Obtener la observación actual del simulador.

        Returns:
            np.ndarray: Vector de observación: [quat(4), angular_vel(3), mag_field(3)]
        """
        try:
            # obtener estado del simulador de rotación
            quaternion = self.rotation_sim.quaternion.copy()
            angular_velocity = self.rotation_sim.angular_velocity.copy()

            # obtener info del campo magnetico
            try:
                # Call the method directly instead of sending a request
                mag_field_data = self.magnetic_sim.magnetic_field
                # extraer componentes x, y, z y convertir de nT a T
                mag_field_inertial = np.array([
                    mag_field_data[0], # north
                    mag_field_data[1], # east
                    mag_field_data[2]  # vertical
                ]) * 1e-9

                # rotar campo magnetico de inercial a cuerpo usando quaternion
                mag_field_body = self._rotate_vector_by_quaternion(mag_field_inertial, quaternion)

            except Exception as e:
                print(f"Warning: Could not get magnetic field: {e}")
                mag_field_body = np.zeros(3)

            observation = np.concatenate([
                quaternion,
                angular_velocity,
                mag_field_body
            ]).astype(np.float32)

            return observation

        except Exception as e:
            print(f"Error getting observation: {e}")
            # retornar observacion default en caso de fallo
            return np.zeros(10, dtype=np.float32)

    def _rotate_vector_by_quaternion(self, vector, quaternion):
        """
        Rotar un vector del marco inercial al marco del cuerpo usando un quaternion.
        Wrapper de lo que ya existe en RotationSimulation.

        Args:
            vector (np.ndarray): Vector 3d en el marco inercial
            quaternion (np.ndarray): Quaternion [qx, qy, qz, qw]

        Returns:
            np.ndarray: Vector rotado en el marco del cuerpo
        """
        try:
            # representar vector como un quaternion puro
            v_quat = np.array([vector[0], vector[1], vector[2], 0.0])

            # conjugado del quaternion (inercial a cuerpo)
            q_conj = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])

            # Usar el método estático existente para la multiplicación de quaterniones
            # rotated_v = q_conj * v_quat * q
            temp = RotationSimulation.quat_mut(q_conj, v_quat)
            rotated_v = RotationSimulation.quat_mut(temp, quaternion)

            # retornar solo la parte del vector
            return rotated_v[:3]

        except Exception as e:
            print(f"Warning: Quaternion rotation failed: {e}")
            return vector

    def _calculate_reward(self, action):
        """
        Calcular la recompensa para el paso actual.

        Args:
            action (np.ndarray): Comando de torque aplicado.

        Returns:
            float: Valor de recompensa
        """
        try:
            # obtener velocidad angular actual
            angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)
        except Exception:
            # si no se puede obtener, se retorna 1
            angular_vel_norm = 1.0

            # obtener "effort" de control
        control_effort = np.linalg.norm(action)

        # funcion de recompensa: penalizar alta velocidad angular y esfuerzo de control
        # puede ser cambiada, requiere experimentación
        reward = -angular_vel_norm - 0.01 * control_effort

        # acá se aplica bonus si es que se logra una velocidad angular muy baja
        if angular_vel_norm < self.success_threshold:
            reward += 10.0

        return reward

    def render(self):
        """
        Renderizar estado actual del entorno.
        """
        if self.render_mode == 'human':
            try:
                quaternion = self.rotation_sim.quaternion
                angular_velocity = self.rotation_sim.angular_velocity
                angular_vel_norm = np.linalg.norm(angular_velocity)

                print(f"Step: {self.current_step:3d} | "
                      f"Time: {self.current_time.timestamp():.4f} | "
                      f"ω_norm: {angular_vel_norm:.4f} rad/s | "
                      f"ω: [{angular_velocity[0]:.3f}, {angular_velocity[1]:.3f}, {angular_velocity[2]:.3f}] rad/s | "
                      f"Episode Reward: {self.episode_reward:.2f} | "
                      f"Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            except Exception as e:
                print(f"Render error: {e}")

        if self.render_mode == 'plot':
            pass

    def close(self):
        """
        Limpiar entorno y reiniciar todos los simuladores externos.
        """
        if self._plot_hist:
            self.show_hist()
        self._stop_simulators()

    def show_hist(self):
        if len(self._observation_hist) == 0:
            print("No hay historial guardado")
            return

        observation_hist = np.array(self._observation_hist)
        quat_hist = observation_hist[:,0:4]
        vel_hist = observation_hist[:,4:7]
        mag_hist = observation_hist[:,7:10]
        torque_hist = observation_hist[:,10:13]

        figure, axes = plt.subplots(3, 1)
        plt.title("Rotation Simulation")
        axes[0].grid(True)
        axes[1].grid(True)
        plt.ion()
        plt.show(block=False)

        axes[0].clear()
        axes[0].plot(self._time_hist, np.array(vel_hist), "--.", label=["x", "y", "z"])
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel('Velocity (rad/s)')
        axes[0].set_xlabel('Time')
        axes[0].grid(True)

        axes[1].clear()
        axes[1].plot(self._time_hist, np.array(quat_hist), "--.", label=["i", "j", "k", "s"])
        axes[1].legend(loc="upper right")
        axes[1].set_ylabel('Quaternion')
        axes[1].set_xlabel('Time')
        axes[1].grid(True)

        axes[2].clear()
        axes[2].plot(self._time_hist, np.array(torque_hist), "--.", label=["Tx", "Ty", "Tz"])
        axes[2].legend(loc="upper right")
        axes[2].set_ylabel('Torque (%)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True)

        plt.show(block=True)


def test_environment_basic():
    """
    Función de prueba para mostrar el funcionamiento del entorno.
    """
    print("=" * 60)
    print("Testing CubeSat Detumbling Environment")
    print("=" * 60)

    env = CubeSatDetumblingEnv(render_mode='human')

    try:
        # reiniciar entorno
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")

        # tomar 10 acciones aleatorias
        for i in range(10):
            action = env.action_space.sample()
            print(f"\nStep {i + 1}: Action = {action}")

            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Reward: {reward:.4f}")

            if terminated or truncated:
                print(f"Episode ended at step {i + 1}")
                if terminated:
                    print("SUCCESS: Detumbling achieved!")
                else:
                    print("Episode truncated (timeout)")
                break

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment test completed!")


if __name__ == "__main__":
    """
    Prueba simple en este mismo script.
    En caso de entrenamiento, se recomienda usar el script train_cubesat_detumbling.py.
    """
    print("=" * 60)
    print("CubeSat Detumbling Environment Test")
    print("=" * 60)

    debug = True  # Activar o desactivar gráficos
    plot_hist = True
    start_time = datetime.fromtimestamp(1758566834)
    time_step = 1
    total_time = 15*60
    granularity = 10

    # crear y probar el entorno
    env = CubeSatDetumblingEnv(render_mode='human', start_time=start_time, time_step=time_step, granularity=granularity,
                               debug=debug, plot_hist=plot_hist)

    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    try:
        episodes = 2
        for i in range(episodes):
            # correr un episodio
            obs, _ = env.reset()
            print(f"\nInitial observation shape: {obs.shape}")
            print("Running N random steps...")

            for step in np.arange(0, total_time, time_step):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    print(f"\nSUCCESS! Episode completed at step {step + 1}")
                    break
                elif truncated:
                    print(f"\nEpisode truncated at step {step + 1}")
                    break

    except Exception as e:
        print(f"Test error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment test completed!")

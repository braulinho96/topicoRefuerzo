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

    def __init__(self, render_mode=None, max_steps=500, start_time=datetime.now(), time_step=0.1, granularity=5, debug=False, plot_hist=False, action_map=None):
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

        self.action_map = action_map if action_map is not None else {
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

        # Registro de la velocidad anterior para bonus de recompenza
        self.prev_angular_vel_norm = None

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
        #self.orbital_sim.update_simulation(current_time)
        #self.magnetic_sim.update_simulation(current_time)

    # Archivo: cubesat_detumbling_rl.py

    def reset(self, seed=None, options=None):
        """
        Función para reiniciar el entorno y comenzar un nuevo episodio.
        ...
        """
        super().reset(seed=seed)

        # parar simulaciones para luego reiniciarlas para nuevo episodio
        self._create_simulators()

        # condiciones iniciales aleatorias o fijas
        initial_angular_velocity = self.np_random.uniform(-1.0, 1.0, size=3)
        initial_quat = self.np_random.normal(size=4)

        # Setear condiciones iniciales
        initial_quat /= np.linalg.norm(initial_quat)
        self.rotation_sim.angular_velocity = initial_angular_velocity
        self.rotation_sim.quaternion = initial_quat

        # reiniciar tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_time = self.start_time
        self.prev_angular_vel_norm = None

        # *** OPTIMIZACIÓN: Calcular y guardar el estado inicial (t_start) de órbita/magnético. ***
        # Actualizar simuladores al tiempo actual (current_time)
        self.orbital_sim.update_simulation(self.current_time)
        self.magnetic_sim.update_simulation(self.current_time)
        
        # Guardar el estado inicial para la interpolación en el primer STEP (USANDO GETTERS)
        self._pos_start = self.orbital_sim.get_position().copy()
        
        # CORRECCIÓN FINAL: Convertir a np.array antes de .copy()
        if hasattr(self.magnetic_sim, 'get_magnetic_field'):
            mag_field_tuple = self.magnetic_sim.get_magnetic_field()
            self._mag_start = np.array(mag_field_tuple).copy() 
        else:
            self._mag_start = self.magnetic_sim.magnetic_field.copy() 

        # La primera observación usa el campo magnético rotado
        mag_field_inertial_T = self._mag_start * 1e-9
        mag_field_body_initial = self._rotate_vector_by_quaternion(mag_field_inertial_T, initial_quat)

        observation = self._get_observation(mag_field_body_initial)
        info = {}

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Ejecuta un solo paso en el entorno dentro de un episodio.
        ...
        """
        # mapear accion discreta a vector de torque
        torque_action = self.action_map[action]

        # aplicar accion de torque al simulador de rotacion
        self.rotation_sim.set_torque(torque_action)

        # *** OPTIMIZACIÓN: Realizar cálculos caros solo 1 vez por step (t_end) ***
        
        time_end = self.current_time + timedelta(seconds=self.time_step)
        
        # 1. Actualizar simuladores al tiempo FINAL (t_end)
        self.orbital_sim.update_simulation(time_end)
        self.magnetic_sim.update_simulation(time_end)
        
        # 2. Capturar el estado 'end' del step usando GETTERS
        self._pos_end = self.orbital_sim.get_position().copy()
        
        # CORRECCIÓN FINAL: Convertir a np.array antes de .copy()
        if hasattr(self.magnetic_sim, 'get_magnetic_field'):
            mag_field_tuple = self.magnetic_sim.get_magnetic_field()
            self._mag_end = np.array(mag_field_tuple).copy()
        else:
            self._mag_end = self.magnetic_sim.magnetic_field.copy()
        
        # Avanzar la simulacion con una granularidad menor
        dt = self.time_step / self.sim_granularity
        
        mag_field_body_final = np.zeros(3) # Para la observación final
        
        for i in range(self.sim_granularity):
            # 1. Avanzar el tiempo y actualizar SÓLO ROTACIÓN
            self.current_time += timedelta(seconds=dt) # Avanzar el tiempo en el step definido
            self._update_simulators(self.current_time) # Actualiza SOLO RotationSimulation

            # 2. INTERPOLACIÓN LINEAL para Magnético
            # Factor de avance lineal dentro del step [0.0, 1.0]
            alpha = (i + 1) / self.sim_granularity
            
            # Interpolación del campo magnético (en nT)
            mag_field_inertial_interp = self._mag_start + alpha * (self._mag_end - self._mag_start)
            
            # Solo guardamos el campo magnético interpolado en la última iteración
            if i == (self.sim_granularity - 1):
                # Rotar el campo magnético (Inercial -> Cuerpo) para la observación final
                quaternion = self.rotation_sim.quaternion.copy()
                
                # Conversión de nT a T antes de la rotación
                mag_field_inertial_T = mag_field_inertial_interp * 1e-9
                mag_field_body_final = self._rotate_vector_by_quaternion(mag_field_inertial_T, quaternion)


            # Guardar historicos para graficar (si está activo)
            if self._debug:
                current_quat = self.rotation_sim.quaternion.copy()
                current_vel = self.rotation_sim.angular_velocity.copy()
                
                obs_hist = np.concatenate([current_quat, current_vel, mag_field_body_final, torque_action])
                self._observation_hist.append(obs_hist)
                self._time_hist.append(self.current_time.timestamp())

        # *** PREPARACIÓN PARA EL PRÓXIMO STEP ***
        self._pos_start = self._pos_end.copy() 
        self._mag_start = self._mag_end.copy()
        
        # obtener nueva observacion (usa mag_field_body_final interpolado)
        observation = self._get_observation(mag_field_body_final)

        # calcular recompensa
        reward = self._calculate_reward(torque_action)
        self.episode_reward += reward

        # revisar si es que termino el episodio
        try:
            angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)
            terminated = bool(angular_vel_norm < self.success_threshold)
        except Exception:
            angular_vel_norm = 1.0
            terminated = False

        # revisar si ocurre un timeout episodico
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

        # funcion de recompensa: penalizar alta velocidad angular y paso de tiempo
        reward = - angular_vel_norm - 0.01*abs(np.linalg.norm(action))

        # acá se aplica bonus si es que se logra una velocidad angular muy baja
        if angular_vel_norm < self.success_threshold:
            reward += 100.0
        
        #Actualizar velocidad anterior
        self.prev_angular_vel_norm = angular_vel_norm

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

    def _get_observation(self, mag_field_body):
        """
        Obtener la observación actual del simulador usando el campo magnético interpolado.

        Returns:
            np.ndarray: Vector de observación: [quat(4), angular_vel(3), mag_field(3)]
        """
        try:
            # obtener estado del simulador de rotación
            quaternion = self.rotation_sim.quaternion.copy()
            angular_velocity = self.rotation_sim.angular_velocity.copy()

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

        except Exception as e:
            print(f"Error getting observation: {e}")
            # retornar observacion default en caso de fallo
            return np.zeros(10, dtype=np.float32)

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

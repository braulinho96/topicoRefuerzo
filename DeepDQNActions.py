import os
import optuna
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from cubesat_detumbling_rl import CubeSatDetumblingEnv

from SatellitePersonality import SatellitePersonality
from Simulations.RotationSimulation import RotationSimulation
import numpy as np
from gymnasium import spaces 

from torch.utils.tensorboard import SummaryWriter

# ============================
# CONFIGURACIONES INICIALES
# ============================

LOG_DIR = "./logs_dqn_detumbling"
MODEL_DIR = "./models_dqn_detumbling"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Forzar uso de GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Dispositivo usado: {device.upper()}")

# ============================
# DEFINICIÓN DE OPTIMIZACIÓN
# ============================

#Para revisar métricas en tiempo real durante el entrenamiento, ejecutar:
#tensorboard --logdir ./logs_dqn_detumbling

class CustomMetricsCallback(BaseCallback):
    """
    Callback personalizado para registrar métricas adicionales en TensorBoard.
    """
    def __init__(self, env, min_torque, mid_torque, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.min_torque = min_torque
        self.mid_torque = mid_torque

    def _on_step(self) -> bool:
        # Registrar la velocidad angular
        try:
            angular_vel_norm = np.linalg.norm(self.env.rotation_sim.angular_velocity)
            self.logger.record("metrics/angular_velocity_norm", angular_vel_norm)
        except Exception as e:
            pass
        
        # Registrar min_torque y mid_torque
        self.logger.record("metrics/min_torque", self.min_torque)
        self.logger.record("metrics/mid_torque", self.mid_torque)
        
        return True


def optimize_dqn(trial):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    Entrena un modelo DQN con hiperparámetros propuestos y devuelve
    la recompensa promedio de evaluación.
    """
    # Definir el rango de optimización de torques
    min_torque = trial.suggest_float("min_torque", 0.01, 0.2) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL  # 1% a 20% del max
    mid_torque = trial.suggest_float("mid_torque", 0.3, 0.7) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL   # 30% a 70% del max

    # Actualizar el mapa de acciones con los valores optimizados
    action_map = {
        0: np.array([SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        1: np.array([-SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        2: np.array([0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        3: np.array([0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        4: np.array([0, 0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        5: np.array([0, 0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        6: np.array([min_torque, 0, 0]),
        7: np.array([-min_torque, 0, 0]),
        8: np.array([0, min_torque, 0]),
        9: np.array([0, -min_torque, 0]),
        10: np.array([0, 0, min_torque]),
        11: np.array([0, 0, -min_torque]),
        12: np.array([mid_torque, 0, 0]),
        13: np.array([-mid_torque, 0, 0]),
        14: np.array([0, mid_torque, 0]),
        15: np.array([0, -mid_torque, 0]),
        16: np.array([0, 0, mid_torque]),
        17: np.array([0, 0, -mid_torque]),
        18: np.array([0, 0, 0]),  # No torque
    }

    # Crear el entorno con el nuevo mapa de acciones
    env = Monitor(CubeSatDetumblingEnv(max_steps=1000, action_map=action_map))

    # Espacio de búsqueda de hiperparámetros
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1000, 5000]),
    }

    # Crear modelo DQN con los hiperparámetros optimizados
    model = DQN(
        "MlpPolicy",
        env,
        **params,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    # Crear el callback personalizado
    callback = CustomMetricsCallback(env, min_torque, mid_torque)

    # Entrenamiento del modelo
    model.learn(total_timesteps=50_000, callback=callback)

    # Evaluar el modelo
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    env.close()
    return mean_reward


# ============================
# OPTIMIZACIÓN BAYESIANA
# ============================

study_name = "dqn_cubesat_optuna"
print("\n🚀 Iniciando optimización con Optuna...")
study = optuna.create_study(direction="maximize", study_name=study_name)
study.optimize(optimize_dqn, n_trials=15)

print("\n✅ Mejor configuración encontrada:")
print(study.best_params)
print(f"Recompensa media: {study.best_value:.2f}")

# ============================
# ENTRENAMIENTO FINAL
# ============================

print("\n🏁 Entrenando modelo final con los mejores hiperparámetros...")

best_params = study.best_params
env = Monitor(CubeSatDetumblingEnv(max_steps=2000))
check_env(env)

# Crear el modelo final con los mejores hiperparámetros
final_model = DQN(
    "MlpPolicy",
    env,
    **best_params,
    verbose=0,
    tensorboard_log=LOG_DIR,
    device=device,
)

# Crear el callback personalizado para el entrenamiento final
final_callback = CustomMetricsCallback(env, 0, 0)

# Entrenamiento principal
final_model.learn(total_timesteps=50_000, callback=final_callback)

# Cerrar SummaryWriter del entrenamiento final
model_path = os.path.join(MODEL_DIR, "dqn_cubesat_optuna_best.zip")
final_model.save(model_path)

print(f"\n💾 Modelo final guardado en: {model_path}")

# ============================
# EVALUACIÓN FINAL
# ============================

mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=10, deterministic=True)
print(f"\n🎯 Recompensa promedio final: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()

print("\n📊 Para ver métricas en tiempo real, ejecuta:")
print("   tensorboard --logdir ./logs_dqn_detumbling")

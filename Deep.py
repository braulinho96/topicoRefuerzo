import os
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from cubesat_detumbling_rl import CubeSatDetumblingEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
from SatellitePersonality import SatellitePersonality
from Simulations.RotationSimulation import RotationSimulation
import numpy as np
from gymnasium import spaces

# ============================
# CONFIGURACIONES INICIALES
# ============================

#Para revisar métricas en tiempo real durante el entrenamiento, ejecutar:
#tensorboard --logdir ./logs_ppo_detumbling

LOG_DIR = "./logs_ppo_detumbling"
MODEL_DIR = "./models_ppo_detumbling"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Forzar uso de GPU si está disponible
device = "cpu"  # PPO funciona mejor en CPU para políticas MLP
print(f"🔥 Dispositivo usado: {device.upper()}")

# ============================
# DEFINICIÓN DE OPTIMIZACIÓN
# ============================

class CustomTensorBoardCallback(BaseCallback):
    """
    Callback que registra métricas personalizadas en el MISMO TensorBoard
    que usa Stable-Baselines3 (train/loss, exploration_rate, etc.).
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            # Acceder al entorno real (DummyVecEnv -> envs[0] -> unwrapped)
            original_env = self.training_env.envs[0].unwrapped
            angular_velocity = original_env.rotation_sim.angular_velocity

            # Métricas físicas
            angular_vel_norm = float(np.linalg.norm(angular_velocity))

            # ===== Métricas personalizadas =====
            self.logger.record("custom/angular_velocity_norm", angular_vel_norm)
            self.logger.record("custom/angular_velocity_x", float(angular_velocity[0]))
            self.logger.record("custom/angular_velocity_y", float(angular_velocity[1]))
            self.logger.record("custom/angular_velocity_z", float(angular_velocity[2]))

        except Exception:
            # Evitar que el callback rompa el entrenamiento
            pass

        return True

def optimize_ppo(trial):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    Entrena un modelo PPO con hiperparámetros propuestos y devuelve
    la recompensa promedio de evaluación.
    """

    # Definir el rango de optimización de torques
    min_torque = trial.suggest_float("min_torque", 0.001, 0.02) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL  # 0.1% a 2% del max
    mid_torque = trial.suggest_float("mid_torque", 0.05, 0.4) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL   # 5% a 40% del max

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
        18: np.array([0, 0, 0]),  
    }

    # Crear el entorno con el nuevo mapa de acciones
    env = Monitor(CubeSatDetumblingEnv(max_steps=200, action_map=action_map))

    # Espacio de búsqueda de hiperparámetros
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02),
    }

    model = PPO(
        "MlpPolicy",
        env,
        **params,
        verbose=0,
        device=device,
    )

    # Imprimimos los hiperparámetros y torques usados en este trial
    print(f"\n🔧 Trial {trial.number} - Hiperparámetros: {params}, min_torque: {min_torque:.4f}, mid_torque: {mid_torque:.4f}")

    # Entrenamiento del modelo
    model.learn(total_timesteps=5_000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    env.close()
    return mean_reward

# ============================
# OPTIMIZACIÓN BAYESIANA
# ============================

study_name = "ppo_cubesat_optuna"
print("\n🚀 Iniciando optimización con Optuna...")
study = optuna.create_study(direction="maximize", study_name=study_name)
study.optimize(optimize_ppo, n_trials=5)

# Almacenamos los parametros del estudio en un archivo JSON para usarlos en la evaluación y entrenamiento final
metadata = {
    "best_params": study.best_params,
    "action_map_info": {
        "min_torque_factor": study.best_params["min_torque"],
        "mid_torque_factor": study.best_params["mid_torque"],
        "max_torque_constant": SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
    }
}
with open(os.path.join(MODEL_DIR, "best_config.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Mejor configuración encontrada:")
print(study.best_params)
print(f"Recompensa media: {study.best_value:.2f}")


# ============================
# ENTRENAMIENTO FINAL
# ============================

print("\n🏁 Entrenando modelo final con los mejores hiperparámetros...")

best_params = study.best_params.copy()
best_min = study.best_params["min_torque"] * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
best_mid = study.best_params["mid_torque"] * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
best_params.pop("min_torque")
best_params.pop("mid_torque")
action_map = {
        0: np.array([SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        1: np.array([-SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        2: np.array([0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        3: np.array([0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        4: np.array([0, 0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        5: np.array([0, 0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        6: np.array([best_min, 0, 0]),
        7: np.array([-best_min, 0, 0]),
        8: np.array([0, best_min, 0]),
        9: np.array([0, -best_min, 0]),
        10: np.array([0, 0, best_min]),
        11: np.array([0, 0, -best_min]),
        12: np.array([best_mid, 0, 0]),
        13: np.array([-best_mid, 0, 0]),
        14: np.array([0, best_mid, 0]),
        15: np.array([0, -best_mid, 0]),
        16: np.array([0, 0, best_mid]),
        17: np.array([0, 0, -best_mid]),
        18: np.array([0, 0, 0]),
    }


env = Monitor(CubeSatDetumblingEnv(max_steps=200, action_map=action_map))
check_env(env)

final_model = PPO(
    "MlpPolicy",
    env,
    **best_params,
    verbose=0,
    tensorboard_log=LOG_DIR,
    device=device,
)

# Crear el callback personalizado para el entrenamiento final
final_callback = CustomTensorBoardCallback()

# Entrenamiento principal
final_model.learn(total_timesteps=5_000, callback=final_callback, log_interval=1)

# Cerrar SummaryWriter del entrenamiento final
model_path = os.path.join(MODEL_DIR, "ppo_cubesat_optuna_best_.zip")
final_model.save(model_path)

print(f"\n💾 Modelo final guardado en: {model_path}")

# ============================
# EVALUACIÓN FINAL
# ============================

mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=5, deterministic=True)
print(f"\n🎯 Recompensa promedio final: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()

print("\n📊 Para ver métricas en tiempo real, ejecuta:")
print("   tensorboard --logdir ./logs_ppo_detumbling")
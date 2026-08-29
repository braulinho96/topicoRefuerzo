import os
import optuna
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from cubesat_detumbling_rl import CubeSatDetumblingEnv
import json

from SatellitePersonality import SatellitePersonality
import numpy as np

print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

# ============================
# CONFIGURACIONES INICIALES
# ============================

LOG_DIR = "./logs_dqn_detumbling"
MODEL_DIR = "./models_dqn_detumbling"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
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
            original_env = self.training_env.envs[0].unwrapped
            angular_velocity = original_env.rotation_sim.angular_velocity
            angular_vel_norm = float(np.linalg.norm(angular_velocity))

            # ===== Métricas personalizadas =====
            self.logger.record("custom/angular_velocity_norm", angular_vel_norm)
            self.logger.record("custom/angular_velocity_x", float(angular_velocity[0]))
            self.logger.record("custom/angular_velocity_y", float(angular_velocity[1]))
            self.logger.record("custom/angular_velocity_z", float(angular_velocity[2]))
        except Exception:
            pass
        return True

def optimize_dqn(trial):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    Entrena un modelo DQN con hiperparámetros propuestos y devuelve
    la recompensa promedio de evaluación.
    """
    # Definir el rango de optimización de torques
    min_torque = trial.suggest_float("min_torque", 0.001, 0.02) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL 
    mid_torque = trial.suggest_float("mid_torque", 0.05, 0.2) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL   
    action_penalty = trial.suggest_float("action_penalty", 0.0001, 0.1)

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
    env = Monitor(CubeSatDetumblingEnv(max_steps=200, action_map=action_map, action_penalty=action_penalty))

    # Espacio de búsqueda de hiperparámetros
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50_000, 500_000, step=50_000),
        "batch_size_power": trial.suggest_int("batch_size_power", 6, 10),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "train_freq": trial.suggest_int("train_freq", 1, 16),
        "gradient_steps": trial.suggest_int("gradient_steps", 1,8),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
        "target_update_interval": trial.suggest_int("target_update_interval", 500, 10_000),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        "tau": trial.suggest_float("tau", 0.005, 1.0),
        "net_arch": trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256], [128, 128, 128]])
    }

    model_params = params.copy()
    model_params["batch_size"] = 2 ** model_params.pop("batch_size_power")
    model_params["policy_kwargs"] = {"net_arch": model_params.pop("net_arch")}

    # Crear modelo DQN con los hiperparámetros optimizados
    model = DQN(
        "MlpPolicy",
        env,
        **model_params,
        verbose=0,
        device=device,
    )
    print(f"\n🔧 Trial {trial.number} - Hiperparámetros: {params}, min_torque: {min_torque:.4f}, mid_torque: {mid_torque:.4f}, action_penalty: {action_penalty:.4f}")
    
    # Entrenamiento del modelo
    model.learn(total_timesteps=50_000)

    # Evaluar el modelo
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    env.close()
    return mean_reward

# ============================
# OPTIMIZACIÓN BAYESIANA
# ============================

study_name = "dqn_cubesat_optuna_MIO"
print("\n🚀 Iniciando optimización con Optuna...")

study = optuna.create_study(direction="maximize", study_name=study_name)
study.optimize(optimize_dqn, n_trials=50)

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

# Extraer parámetros de entorno y quitarlos del diccionario
best_min = best_params.pop("min_torque") * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
best_mid = best_params.pop("mid_torque") * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
best_penalty = best_params.pop("action_penalty")

# Formatear parámetros que Stable Baselines3 exige empaquetados/procesados
best_params["batch_size"] = 2 ** best_params.pop("batch_size_power")
best_params["policy_kwargs"] = {"net_arch": best_params.pop("net_arch")}

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

env = Monitor(CubeSatDetumblingEnv(max_steps=200, action_map=action_map, action_penalty=best_penalty))
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
final_callback = CustomTensorBoardCallback()

# Entrenamiento principal
final_model.learn(total_timesteps=400_000, callback=final_callback, log_interval=1)

# Cerrar SummaryWriter del entrenamiento final
model_path = os.path.join(MODEL_DIR, "dqn_cubesat_optuna_best_NUEVOS.zip")
final_model.save(model_path)

print(f"\n💾 Modelo final guardado en: {model_path}")

# ============================
# EVALUACIÓN FINAL
# ============================

mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=50, deterministic=True)
print(f"\n🎯 Recompensa promedio final: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()

print("\n📊 Para ver métricas en tiempo real, ejecuta:")
print("   tensorboard --logdir ./logs_dqn_detumbling")

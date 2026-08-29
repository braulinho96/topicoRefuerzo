import os
import optuna
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from cubesat_detumbling_rl import CubeSatDetumblingEnv

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

def optimize_dqn(trial):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    Entrena un modelo DQN con hiperparámetros propuestos y devuelve
    la recompensa promedio de evaluación.
    """
    env = Monitor(CubeSatDetumblingEnv(max_steps=1000))

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

    model = DQN(
        "MlpPolicy",
        env,
        **params,
        verbose=0,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    model.learn(total_timesteps=50_000)
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

final_model = DQN(
    "MlpPolicy",
    env,
    **best_params,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device=device,
)

# Entrenamiento principal
final_model.learn(total_timesteps=50_000)
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

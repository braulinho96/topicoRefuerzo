import os
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from cubesat_detumbling_rl import CubeSatDetumblingEnv

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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Dispositivo usado: {device.upper()}")

# ============================
# DEFINICIÓN DE OPTIMIZACIÓN
# ============================

def optimize_ppo(trial):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    Entrena un modelo PPO con hiperparámetros propuestos y devuelve
    la recompensa promedio de evaluación.
    """
    env = Monitor(CubeSatDetumblingEnv(max_steps=1000))

    # Espacio de búsqueda de hiperparámetros
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 0.99),
        "ent_coef": trial.suggest_uniform("ent_coef", 0.0, 0.02),
    }

    model = PPO(
        "MlpPolicy",
        env,
        **params,
        verbose=0,
        tensorboard_log=LOG_DIR,
        device=device,
    )

    model.learn(total_timesteps=200_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    env.close()
    return mean_reward


# ============================
# OPTIMIZACIÓN BAYESIANA
# ============================

study_name = "ppo_cubesat_optuna"
print("\n🚀 Iniciando optimización con Optuna...")
study = optuna.create_study(direction="maximize", study_name=study_name)
study.optimize(optimize_ppo, n_trials=15)

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

final_model = PPO(
    "MlpPolicy",
    env,
    **best_params,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device=device,
)

final_model.learn(total_timesteps=600_000)
model_path = os.path.join(MODEL_DIR, "ppo_cubesat_optuna_best.zip")
final_model.save(model_path)

print(f"\n💾 Modelo final guardado en: {model_path}")

# ============================
# EVALUACIÓN FINAL
# ============================

mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=10, deterministic=True)
print(f"\n🎯 Recompensa promedio final: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()

print("\n📊 Para ver métricas en tiempo real, ejecuta:")
print("   tensorboard --logdir ./logs_ppo_detumbling")
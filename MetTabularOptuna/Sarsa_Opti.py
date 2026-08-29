import os
import optuna
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from cubesat_detumbling_rl import CubeSatDetumblingEnv
import json
import numpy as np
import pickle
from Metodos_Tabulares import discretize_state, train_sarsa_td

from SatellitePersonality import SatellitePersonality
from Simulations.RotationSimulation import RotationSimulation
import numpy as np
from gymnasium import spaces

print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

# ============================
# CONFIGURACIONES INICIALES
# ============================

LOG_DIR = "./logs_sarsa_detumbling"
MODEL_DIR = "./models_sarsa_detumbling"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Dispositivo usado: {device.upper()}")

# ============================
# DEFINICIÓN DE OPTIMIZACIÓN
# ============================

def optimize_sarsa(trial):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    Entrena un agente SARSA con hiperparámetros propuestos y devuelve
    la recompensa promedio de evaluación.
    """
    # Rango para diferentes torques
    min_torque = trial.suggest_float("min_torque", 0.01, 0.2) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
    mid_torque = trial.suggest_float("mid_torque", 0.3, 0.7) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
    micro_torque = trial.suggest_float("micro_torque", 0.001, 0.01) * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL

    # Actualizar el mapa de acciones con los valores optimizados
    action_map = {
        0: np.array([SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        1: np.array([-SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
        2: np.array([0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        3: np.array([0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
        4: np.array([0, 0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        5: np.array([0, 0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
        6: np.array([0,0,0]),
    }




















    # Crear el entorno con el nuevo mapa de acciones
    env = CubeSatDetumblingEnv(max_steps=200, action_map=action_map, fast_mode=True)

    # Espacio de búsqueda de hiperparámetros
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "discount_factor": trial.suggest_float("discount_factor", 0.8, 0.999),
        "epsilon": trial.suggest_float("epsilon", 0.5, 1.0),
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.9, 0.9999),
        "min_epsilon": trial.suggest_float("min_epsilon", 0.001, 0.1),
    }

    
    #print(f"\n🔧 Trial {trial.number} - Hiperparámetros: {params}, min_torque: {min_torque:.4f}, mid_torque: {mid_torque:.4f}")

    # Entrenamiento del agente SARSA
    q_table, ang_vel_bins, rewards_per_episode = train_sarsa_td(
        episodes=5000,  
        max_steps=400,
        fast_mode=True,
        granularity=5,
        numero_bins=15,  
        plot_results=False,  
        **params
    )

    # Evaluar el agente entrenado
    
    eval_rewards = []
    for _ in range(50):
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)
        done = False
        total_reward = 0
        step_count = 0
        while not done and step_count < 200:
            action = np.argmax(q_table[state, :])
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize_state(obs, ang_vel_bins)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        eval_rewards.append(total_reward)

    mean_reward = np.mean(eval_rewards)
    env.close()
    return mean_reward

# ============================
# OPTIMIZACIÓN BAYESIANA
# ============================

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=3,
    interval_steps=1,
)

study_name = "sarsa_cubesat_optuna_MIO"
print("\n🚀 Iniciando optimización con Optuna...")
study = optuna.create_study(direction="maximize", study_name=study_name, pruner=pruner)
study.optimize(optimize_sarsa, n_trials=50)  # Para tabular suele siempre encontrar el mejor en pocos trials con 50 deberia ser suficiente

# Parametros de Optuna en un JSON para la evaluación y entrenamiento final
metadata = {
    "best_params": study.best_params,
    "action_map_info": {
        "max_torque_constant": SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
    }
}
with open(os.path.join(MODEL_DIR, "best_config_sarsa.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Mejor configuración encontrada:")
print(study.best_params)
print(f"Recompensa media: {study.best_value:.2f}")

# ============================
# ENTRENAMIENTO FINAL
# ============================

print("\n🏁 Entrenando modelo final con los mejores hiperparámetros...")

best_params = study.best_params.copy()
#best_min = study.best_params["min_torque"] * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
#best_mid = study.best_params["mid_torque"] * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
#best_micro = study.best_params["micro_torque"] * SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
#best_params.pop("min_torque")
#best_params.pop("mid_torque")
#best_params.pop("micro_torque")
action_map = {
    0: np.array([SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
    1: np.array([-SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0, 0]),
    2: np.array([0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
    3: np.array([0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL, 0]),
    4: np.array([0, 0, SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
    5: np.array([0, 0, -SatellitePersonality.MAX_TORQUE_REACTION_WHEEL]),
    6: np.array([0, 0, 0]),
}

env = CubeSatDetumblingEnv(max_steps=500, action_map=action_map, fast_mode=True)

# Entrenamiento final con mejores hiperparámetros
q_table_final, ang_vel_bins_final, rewards_final = train_sarsa_td(
    episodes=10000,  
    max_steps=400,
    fast_mode=True,
    granularity=5,
    numero_bins=15,  
    plot_results=True,  
    **best_params
)


model_data = {
    "q_table": q_table_final,
    "ang_vel_bins": ang_vel_bins_final,
    "action_map": action_map,
    "best_params": study.best_params
}

with open(os.path.join(MODEL_DIR, "sarsa_cubesat_optuna_best.pkl"), "wb") as f:
    pickle.dump(model_data, f)

#Pasa algo raro y el optuna best no tiene nada guardado si no el otro pkl que se crea si
print(f"\n💾 Modelo final guardado en: {os.path.join(MODEL_DIR, 'sarsa_cubesat_optuna_best.pkl')}")

env.close()

print("\n📊 Optimización completada. Los mejores parámetros se guardaron en 'best_config_sarsa.json'")

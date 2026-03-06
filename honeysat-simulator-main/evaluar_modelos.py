import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from cubesat_detumbling_rl import CubeSatDetumblingEnv
from SatellitePersonality import SatellitePersonality

# ============================
# CONFIGURACIÓN DE RUTAS
# ============================
print("1 Para DQN | 2 Para PPO")
Choice = input("Ingresa tu elección (1 o 2): ")

if Choice == "1":
    MODEL_DIR = "./models_dqn_detumbling"
    MODEL_PATH = os.path.join(MODEL_DIR, "dqn_cubesat_optuna_best_NUEVOS.zip") 
    CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")
else:
    MODEL_DIR = "./models_ppo_detumbling"
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_cubesat_optuna_best_.zip")
    CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")

    
def load_best_config(config_path):
    """Lee el JSON, reconstruye el action_map y recupera la penalización."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    factors = config["action_map_info"]
    max_t = factors["max_torque_constant"]
    min_t = factors["min_torque_factor"] * max_t
    mid_t = factors["mid_torque_factor"] * max_t

    action_map = {
        0: np.array([max_t, 0, 0]),  1: np.array([-max_t, 0, 0]),
        2: np.array([0, max_t, 0]),  3: np.array([0, -max_t, 0]),
        4: np.array([0, 0, max_t]),  5: np.array([0, 0, -max_t]),
        6: np.array([min_t, 0, 0]),  7: np.array([-min_t, 0, 0]),
        8: np.array([0, min_t, 0]),  9: np.array([0, -min_t, 0]),
        10: np.array([0, 0, min_t]), 11: np.array([0, 0, -min_t]),
        12: np.array([mid_t, 0, 0]), 13: np.array([-mid_t, 0, 0]),
        14: np.array([0, mid_t, 0]), 15: np.array([0, -mid_t, 0]),
        16: np.array([0, 0, mid_t]), 17: np.array([0, 0, -mid_t]),
        18: np.array([0, 0, 0]),
    }
    
    # Recuperamos el penalty (usamos 0.0 por defecto si no existe en el JSON)
    action_penalty = config.get("action_penalty", 0.0)
    
    return action_map, action_penalty


def evaluate_success_rate(model, env, n_episodes=100, threshold=0.01):
    """Calcula la tasa de éxito, promedio de pasos y estadísticas de recompensa."""
    success_count = 0
    
    # Listas para guardar las métricas de cada episodio
    episode_rewards = []
    episode_steps = []
    
    print(f"\n🧐 Evaluando {n_episodes} episodios de forma silenciosa...")
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        
        # Variables temporales para el episodio actual
        current_ep_reward = 0.0
        current_ep_steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action) # Convertir a entero de forma segura
            
            # Extraemos la recompensa (reward) también
            obs, reward, done, truncated, _ = env.step(action)
            
            current_ep_reward += reward
            current_ep_steps += 1
        
        # Guardamos los resultados al terminar el episodio
        episode_rewards.append(current_ep_reward)
        episode_steps.append(current_ep_steps)
        
        # Comprobar éxito físico
        final_vel = np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity)
        if final_vel < threshold:
            success_count += 1

    # --- CÁLCULOS FINALES ---
    success_rate = (success_count / n_episodes) * 100
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_steps = np.mean(episode_steps)

    # --- IMPRIMIR RESULTADOS ---
    print("\n" + "="*40)
    print(" 📊 RESULTADOS DE LA EVALUACIÓN 📊")
    print("="*40)
    print(f"✅ Tasa de Éxito Final:      {success_rate:.2f}%")
    print(f"👣 Promedio de Pasos:        {mean_steps:.2f} pasos")
    print(f"💰 Recompensa Promedio:      {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*40 + "\n")


def debug_and_plot_episode(model, env, max_steps=200):
    """Ejecuta un solo episodio, imprime la tabla paso a paso y genera el gráfico de 2 paneles."""
    print("\n" + "="*80)
    print(f"{'PASO':<6} | {'ACCIÓN':<10} | {'VELOCIDAD ANGULAR (rad/s)':<30} | {'NORMA':<10} | {'ESTADO'}")
    print("="*80)

    obs, _ = env.reset()
    done, truncated = False, False
    step = 0
    
    history_steps = []
    history_vel_norm = []
    history_torque_norm = []
    
    threshold = env.unwrapped.success_threshold
    prev_vel = None

    while not (done or truncated) and step < max_steps:
        # 1. Predicción
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        torque = env.unwrapped.action_map[action]
        torque_mag = np.linalg.norm(torque)
        
        # 2. Datos físicos
        current_vel = env.unwrapped.rotation_sim.angular_velocity.copy()
        norm = np.linalg.norm(current_vel)
        
        history_steps.append(step)
        history_vel_norm.append(norm)
        history_torque_norm.append(torque_mag)
        
        # 3. Ejecutar paso
        obs, reward, done, truncated, info = env.step(action)
        
        # 4. Diagnóstico de oscilación
        oscillation_warning = ""
        if prev_vel is not None:
            if any((current_vel * prev_vel) < 0):
                oscillation_warning = "⚠️ REBOTE"

        # 5. Imprimir consola
        vel_str = f"[{current_vel[0]:.4f}, {current_vel[1]:.4f}, {current_vel[2]:.4f}]"
        status = "🎯 ÉXITO" if done else ("⏱️ LÍMITE" if truncated else oscillation_warning)
        print(f"{step:<6} | {action:<10} | {vel_str:<30} | {norm:.5f} | {status}")
        
        prev_vel = current_vel
        step += 1

    print("="*80)

    # --- GENERACIÓN DEL GRÁFICO DE DIAGNÓSTICO ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(history_steps, history_vel_norm, label='Velocidad Angular (Norma)', color='#1f77b4', marker='o', markersize=3)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral de Éxito ({threshold})')
    ax1.set_ylabel('Rad/s')
    ax1.set_title(f'Análisis de Estabilidad del Satélite (Episodio de {step} pasos)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.fill_between(history_steps, history_torque_norm, color='green', alpha=0.2)
    ax2.step(history_steps, history_torque_norm, where='post', color='green', label='Esfuerzo de Torque')
    ax2.set_ylabel('N·m (Magnitud)')
    ax2.set_xlabel('Paso de Tiempo (1 step = 1s)')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if os.path.exists(CONFIG_PATH) and os.path.exists(MODEL_PATH):
        
        # 1. Recuperar Action Map y Penalty
        action_map, action_penalty = load_best_config(CONFIG_PATH)
        
        # 2. Crear Entorno idéntico al entrenamiento
        env = Monitor(CubeSatDetumblingEnv(max_steps=200, action_map=action_map, action_penalty=action_penalty))
        
        # 3. Cargar Modelo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Cargando modelo en: {device.upper()}")
        
        if Choice == "1":
            model = DQN.load(MODEL_PATH, env=env, device=device)
        else:
            model = PPO.load(MODEL_PATH, env=env, device=device)
            
        print("🚀 Modelo y configuración cargados correctamente.")

        # 4. Evaluar estadística general (ej. 1000 episodios)
        evaluate_success_rate(model, env, n_episodes=10000)

        # 5. Debug visual paso a paso (1 episodio)
        debug_and_plot_episode(model, env)

        env.close() 
    else:
        print(f"❌ Error: No se encontraron los archivos.\nBuscando modelo en: {MODEL_PATH}\nBuscando config en: {CONFIG_PATH}")
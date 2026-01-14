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
print(f" 1 Pa DQN 2 PPO")
Choice = input("Enter your name: ")
if Choice == "1":
    MODEL_DIR = "./models_dqn_detumbling"
    MODEL_PATH = os.path.join(MODEL_DIR, "dqn_cubesat_optuna_best_.zip")
    CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")
else:
    MODEL_DIR = "./models_ppo_detumbling"
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_cubesat_optuna_best_.zip")
    CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")
def load_best_action_map(config_path):
    """Lee el JSON y reconstruye el action_map usado en el entrenamiento."""
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
    return action_map

def evaluate_and_plot(model, env, n_episodes=100, threshold=0.01):
    """Calcula tasa de éxito y genera gráfica de un episodio."""
    success_count = 0
    
    print(f"🧐 Evaluando {n_episodes} episodios...")
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            obs, _, done, truncated, _ = env.step(action)
        
        # Comprobar éxito físico
        final_vel = np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity)
        if final_vel < threshold:
            success_count += 1

    print(f"✅ Tasa de Éxito Final: {(success_count/n_episodes)*100}%")

    # --- Generar Gráfica de un episodio ---
    obs, _ = env.reset()
    done, truncated = False, False
    history = [np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity)]
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = action.item()
        obs, _, done, truncated, _ = env.step(action)
        history.append(np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity))

    plt.figure(figsize=(10, 5))
    plt.plot(history, label="Velocidad Angular (rad/s)", color="blue")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Umbral Éxito")
    plt.title("Frenado del Satélite - Modelo Optimizado")
    plt.xlabel("Pasos")
    plt.ylabel("||ω||")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def debug_and_plot_episode(model, env, max_steps=200):
    print("\n" + "="*80)
    print(f"{'PASO':<6} | {'ACCIÓN':<10} | {'VELOCIDAD ANGULAR (rad/s)':<30} | {'NORMA':<10} | {'ESTADO'}")
    print("="*80)

    obs, _ = env.reset()
    done, truncated = False, False
    step = 0
    
    # Listas para guardar datos del gráfico
    history_steps = []
    history_vel_norm = []
    history_torque_norm = []
    
    threshold = env.unwrapped.success_threshold
    prev_vel = None

    while not (done or truncated) and step < max_steps:
        # 1. Predicción y extracción de torque
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        torque = env.unwrapped.action_map[action]
        torque_mag = np.linalg.norm(torque)
        
        # 2. Datos físicos antes del paso
        current_vel = env.unwrapped.rotation_sim.angular_velocity.copy()
        norm = np.linalg.norm(current_vel)
        
        # Guardar para el gráfico
        history_steps.append(step)
        history_vel_norm.append(norm)
        history_torque_norm.append(torque_mag)
        
        # 3. Ejecutar paso en el simulador
        obs, reward, done, truncated, info = env.step(action)
        
        # 4. Diagnóstico de oscilación (cambio de signo)
        oscillation_warning = ""
        if prev_vel is not None:
            if any((current_vel * prev_vel) < 0):
                oscillation_warning = "⚠️ REBOTE"

        # 5. Print por consola
        vel_str = f"[{current_vel[0]:.4f}, {current_vel[1]:.4f}, {current_vel[2]:.4f}]"
        status = "🎯 ÉXITO" if done else ("⏱️ LÍMITE" if truncated else oscillation_warning)
        print(f"{step:<6} | {action:<10} | {vel_str:<30} | {norm:.5f} | {status}")
        
        prev_vel = current_vel
        step += 1

    print("="*80)

    # --- GENERACIÓN DEL GRÁFICO DE DIAGNÓSTICO ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel Superior: Velocidad y Umbral
    ax1.plot(history_steps, history_vel_norm, label='Velocidad Angular (Norma)', color='#1f77b4', marker='o', markersize=3)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral de Éxito ({threshold})')
    ax1.set_ylabel('Rad/s')
    ax1.set_title(f'Análisis de Estabilidad del Satélite (Episodio de {step} pasos)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    # Panel Inferior: Torque aplicado
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
        # 1. Recuperar Action Map
        action_map = load_best_action_map(CONFIG_PATH)
        
        # 2. Crear Entorno idéntico al entrenamiento
        env = Monitor(CubeSatDetumblingEnv(max_steps=200, action_map=action_map))
        
        # 3. Cargar Modelo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if Choice == "1":
            model = DQN.load(MODEL_PATH, env=env, device=device)
        else:
            model = PPO.load(MODEL_PATH, env=env, device="cpu")
        print("🚀 Modelo y configuración cargados correctamente.")

        # 4. Evaluar
        evaluate_and_plot(model, env)

        # 4. Debug paso a paso
        #debug_and_plot_episode(model, env)

        env.close()
    else:
        print("❌ Error: Faltan archivos .zip o .json en la carpeta de modelos.")
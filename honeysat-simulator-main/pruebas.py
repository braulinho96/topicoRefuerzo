import os
import json
import torch
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from cubesat_detumbling_rl import CubeSatDetumblingEnv
from SatellitePersonality import SatellitePersonality

# ============================
# CONFIGURACIÓN
# ============================
MODEL_DIR = "./models_dqn_detumbling" # Cambia a ppo si es necesario
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_cubesat_optuna_best_NUEVOS.zip")
CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")
N_EPISODES = 1000 # Episodios por cada prueba

def load_base_config():
    with open(CONFIG_PATH, "r") as f:
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
    penalty = config.get("action_penalty", 0.0)
    return action_map, penalty

def run_stress_test(model, action_map, penalty, scenario_name, incertidumbre_pct=0.0, actuator_efficiency=1.0, extreme_spin=False):
    """Ejecuta episodios de prueba inyectando perturbaciones específicas."""
    
    # Aplicar degradación a los actuadores modificando el action map
    degraded_action_map = {k: v * actuator_efficiency for k, v in action_map.items()}
    
    env = Monitor(CubeSatDetumblingEnv(max_steps=500, action_map=degraded_action_map, action_penalty=penalty))
    threshold = env.unwrapped.success_threshold
    
    success_count = 0
    episode_rewards = []
    episode_steps = []
    
    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        
        # Inyectar rotación extrema (sobrescribir el reset por defecto del entorno)
        if extreme_spin:
            env.unwrapped.rotation_sim.angular_velocity = np.random.uniform(-2.0, 2.0, size=3)
            # Recalculamos la observación inicial tras el cambio brusco
            mag_field_inertial_T = env.unwrapped._mag_start * 1e-9
            quat = env.unwrapped.rotation_sim.quaternion
            mag_field_body = env.unwrapped._rotate_vector_by_quaternion(mag_field_inertial_T, quat)
            obs = env.unwrapped._get_observation(mag_field_body)

        done, truncated = False, False
        ep_reward = 0.0
        ep_steps = 0
        
        while not (done or truncated):
            # 1. Copia de la observación para NO corromper el entorno real
            obs_neurona = obs.copy()
            
            # 2. Inyectar ruido PROPORCIONAL (Incertidumbre) en el giroscopio
            if incertidumbre_pct > 0:
                # Extraemos la velocidad angular real (índices 4 al 6)
                w_actual = obs_neurona[4:7]
                
                # Ruido proporcional: Normal centrada en 1.0 con desviación = incertidumbre_pct
                factor_ruido = np.random.normal(loc=1.0, scale=incertidumbre_pct, size=3)
                
                # Ruido de piso (blanco): Simula el error térmico fijo del sensor (0.0001 rad/s)
                ruido_piso = np.random.normal(loc=0.0, scale=0.0001, size=3)
                
                # Aplicamos ambos ruidos EXCLUSIVAMENTE a las velocidades
                obs_neurona[4:7] = (w_actual * factor_ruido) + ruido_piso
                
            # 3. La neurona predice usando la observación RUIDOSA (obs_neurona)
            action, _ = model.predict(obs_neurona, deterministic=True)
            
            # 4. El entorno ejecuta el paso y actualiza la observación PERFECTA (obs)
            obs, reward, done, truncated, _ = env.step(int(action))
            
            ep_reward += reward
            ep_steps += 1
            
        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        
        final_vel = np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity)
        if final_vel < threshold:
            success_count += 1
            
    env.close()
    
    return {
        "scenario": scenario_name,
        "success_rate": (success_count / N_EPISODES) * 100,
        "mean_steps": np.mean(episode_steps),
        "mean_reward": np.mean(episode_rewards)
    }

if __name__ == "__main__":
    print("⚙️ Preparando Entorno y Cargando Modelo...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_action_map, base_penalty = load_base_config()
    env_dummy = Monitor(CubeSatDetumblingEnv(max_steps=500, action_map=base_action_map, action_penalty=base_penalty))
    
    # Cambia DQN por PPO si vas a evaluar ese modelo
    model = DQN.load(MODEL_PATH, env=env_dummy, device=device)
    env_dummy.close()
    
    print(f"🚀 Iniciando Test de Generalización ({N_EPISODES} episodios por escenario)\n")
    
    results = []
    
    # 1. Escenario Base (Ideal)
    print("▶ Evaluando Escenario Base...")
    results.append(run_stress_test(model, base_action_map, base_penalty, "Base (Simulación Ideal)"))
    
    # 2. Ruido de Sensores (5% de incertidumbre + ruido térmico)
    print("▶ Evaluando Ruido de Sensores...")
    results.append(run_stress_test(model, base_action_map, base_penalty, "Incertidumbre Giroscopio (5%)", incertidumbre_pct=0.05))
    
    # 3. Degradación de Actuadores (70% potencia)
    print("▶ Evaluando Degradación Actuadores...")
    results.append(run_stress_test(model, base_action_map, base_penalty, "Actuadores al 70%", actuator_efficiency=0.7))
    
    # 4. Rotación Inicial Extrema (-2.0 a 2.0 rad/s)
    print("▶ Evaluando Rotación Extrema...")
    results.append(run_stress_test(model, base_action_map, base_penalty, "Rotación Extrema (2x)", extreme_spin=True))
    
    # 5. Caos Total (Todos los problemas juntos)
    print("▶ Evaluando Caos Total...")
    results.append(run_stress_test(model, base_action_map, base_penalty, "Modo Pesadilla (Todo)", incertidumbre_pct=0.05, actuator_efficiency=0.7, extreme_spin=True))
    
    # --- IMPRIMIR TABLA DE RESULTADOS ---
    print("\n" + "="*85)
    print(f"{'ESCENARIO':<30} | {'TASA DE ÉXITO':<15} | {'PASOS PROMEDIO':<15} | {'RECOMPENSA'}")
    print("="*85)
    for r in results:
        success_str = f"{r['success_rate']:.1f}%"
        steps_str = f"{r['mean_steps']:.1f}"
        rew_str = f"{r['mean_reward']:.2f}"
        print(f"{r['scenario']:<30} | {success_str:<15} | {steps_str:<15} | {rew_str}")
    print("="*85)
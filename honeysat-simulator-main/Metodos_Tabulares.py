
import numpy as np
from cubesat_detumbling_rl import CubeSatDetumblingEnv


def run_random_agent(episodes=10, max_steps=100):
    """
    Ejecuta un agente aleatorio en el entorno CubeSatDetumblingEnv.
    Registra y muestra el desempeño promedio y la variabilidad de las recompensas.
    """
    env = CubeSatDetumblingEnv(render_mode=None, max_steps=max_steps)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action = env.action_space.sample()  # Acción aleatoria
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total_reward)
        print(f"Episodio {ep+1}: Recompensa total = {total_reward:.2f}, Pasos = {steps}")
    env.close()
    print("\nResumen agente aleatorio:")
    print(f"Recompensa promedio: {np.mean(rewards):.2f}")
    print(f"Recompensa mínima: {np.min(rewards):.2f}")
    print(f"Recompensa máxima: {np.max(rewards):.2f}")
    print(f"Desviación estándar: {np.std(rewards):.2f}")
    return rewards

if __name__ == "__main__":
    print("🛰️  CUBESAT DETUMBLING RL DEMO WITH Q-LEARNING")
    print()
    
    
    print("\n--- Evaluación agente aleatorio ---")
    run_random_agent(episodes=10, max_steps=100)
    print("\n🎉 Demo completada!")

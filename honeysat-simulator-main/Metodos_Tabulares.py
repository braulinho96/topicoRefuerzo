
    
import matplotlib.pyplot as plt
import numpy as np
import pickle
from cubesat_detumbling_rl import CubeSatDetumblingEnv

def discretize_state(observation, bins):
    """Discretiza una observación continua en un único entero."""
    discretized = []
    # Discretiza la velocidad angular (observation[4:7])
    for i in range(4, 7):
        # Recorta la observación para que esté dentro del rango de los bins
        clipped_obs = np.clip(observation[i], bins[i-4][0], bins[i-4][-1])
        digit = np.digitize(clipped_obs, bins[i-4]) - 1                         # -1 para que los índices comiencen en 0
        discretized.append(digit)
    
    # Combina las partes discretizadas en un único índice de estado
    # Esto da 10*10*10 = 1000 estados discretos
    return int(sum(d * (len(b) - 1)**i for i, (d, b) in enumerate(zip(discretized, bins))))

def run_random_agent(episodes=10, max_steps=100):
    """
    Ejecuta un agente aleatorio en el entorno CubeSatDetumblingEnv.
    Registra y muestra el desempeño promedio y la variabilidad de las recompensas.
    """
    print("=" * 50)
    print("🎲 INICIANDO EVALUACIÓN DEL AGENTE ALEATORIO")
    print(f"🔬 Episodios de evaluación: {episodes}")
    print("=" * 50)

    env = CubeSatDetumblingEnv(render_mode=None, max_steps=max_steps)
    total_rewards = []
    mumero_bins = 10
    ang_vel_bins = [
        np.linspace(-1.5, 1.5, mumero_bins),
        np.linspace(-1.5, 1.5, mumero_bins),
        np.linspace(-1.5, 1.5, mumero_bins)
        ]
    num_states = mumero_bins**3
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action = env.action_space.sample() # Elige acción aleatoria
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        total_rewards.append(total_reward)
        print(f"Episodio {ep+1}: Recompensa total = {total_reward:.2f}, Pasos = {steps}")
    env.close()
    return q_table, ang_vel_bins, total_rewards

def train_q_learning(episodes=10, max_steps=500, alpha=0.1, gamma=0.7, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    """Entrena un agente de Q-learning en el entorno de detumbling del CubeSat."""
    print("=" * 50)
    print("🚀 INICIANDO ENTRENAMIENTO DE Q-LEARNING")
    print(f"📈 Episodios totales: {episodes:,}")
    print("=" * 50)

    env = CubeSatDetumblingEnv(max_steps=max_steps)

    # Bins de discretización para la velocidad angular (3 dimensiones)
    ang_vel_bins = [
        np.linspace(-1.5, 1.5, 10),
        np.linspace(-1.5, 1.5, 10),
        np.linspace(-1.5, 1.5, 10)
    ]
    numero_bins = 10
    num_states = numero_bins**3
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            # Eleccion Epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q_table[state, :])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, ang_vel_bins)
            done = terminated or truncated
            total_reward += reward
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state
            step_count += 1
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(total_reward)
        print(f"Episodio {episode+1}/{episodes}: Recompensa total = {total_reward:.2f}, Pasos = {step_count}")

    env.close()
    return q_table, ang_vel_bins, total_rewards

def train_monte_carlo(episodes=10, max_steps=500, gamma=0.7, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    """Entrena un agente de Monte Carlo en el entorno de detumbling del CubeSat."""
    print("=" * 50)
    print("💰 INICIANDO ENTRENAMIENTO DE MONTE CARLO")
    print(f"📈 Episodios totales: {episodes:,}")
    print("=" * 50)

    env = CubeSatDetumblingEnv(max_steps=max_steps)
    ang_vel_bins = [
        np.linspace(-1.5, 1.5, 10),
        np.linspace(-1.5, 1.5, 10),
        np.linspace(-1.5, 1.5, 10)
    ]
    numero_bins = 10
    num_states = numero_bins**3
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    returns = {(s, a): [] for s in range(num_states) for a in range(num_actions)}
    total_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)
        done = False
        trajectory = []
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, ang_vel_bins)
            
            trajectory.append({'state': state, 'action': action, 'reward': reward})
            total_reward += reward
            state = next_state
            step_count += 1
            done = terminated or truncated
        
        # Actualización de Monte Carlo: Iterar a través de la trayectoria al final del episodio
        G = 0
        for step in reversed(trajectory):
            # Calcular la recompensa acumulada
            G = gamma * G + step['reward']
            state = step['state']
            action = step['action']
            
            # Solo actualiza la primera vez que se visita el par (estado, acción)
            if (state, action) not in [(s['state'], s['action']) for s in trajectory[:trajectory.index(step)]]:
                returns[(state, action)].append(G)
                q_table[state, action] = np.mean(returns[(state, action)])

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(total_reward)
        print(f"Episodio {episode+1}/{episodes}: Recompensa total = {total_reward:.2f}, Pasos = {step_count}")

    env.close()
    
    return q_table, ang_vel_bins, total_rewards

def evaluate_agent(q_table, ang_vel_bins, episodes=10, max_steps=100, agent_name="Q-Learning"): 
    """Evalúa un agente entrenado."""
    print("\n" + "=" * 50)
    print(f"📈 INICIANDO EVALUACIÓN DEL AGENTE {agent_name.upper()}")
    print(f"🔬 Episodios de evaluación: {episodes}")
    print("=" * 50)

    eval_env = CubeSatDetumblingEnv(max_steps=max_steps)
    total_rewards = []
    success_count = 0

    for episode in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            state = discretize_state(obs, ang_vel_bins)
            action = np.argmax(q_table[state, :])
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        total_rewards.append(total_reward)

        if terminated:
            success_count += 1
            print(f"  ✅ ÉXITO! Detumbling logrado en {step_count} pasos. Recompensa: {total_reward:.2f}")
        else:
            print(f"  ❌ Episodio finalizado después de {step_count} pasos. Recompensa: {total_reward:.2f}")

    eval_env.close()

    print("\n" + "=" * 50)
    print(f"📊 RESUMEN DE EVALUACIÓN ({agent_name.upper()})")
    print(f"🏆 Tasa de éxito: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"📈 Recompensa promedio: {np.mean(total_rewards):.2f}")
    print(f"📉 Recompensa máxima: {np.max(total_rewards):.2f}")
    print(f"📉 Recompensa mínima: {np.min(total_rewards):.2f}")
    print(f"📊 Desviación estándar de la recompensa: {np.std(total_rewards):.2f}")
    print("=" * 50)
    return total_rewards

if __name__ == "__main__":
    print("🛰️  DEMO DE AGENTE DE APRENDIZAJE POR REFUERZO DE DETUMBLING")
    print()
    
    # 1. Evaluación del agente aleatorio (línea de base)
    q_table_random, ang_vel_bins_random, rewards_random = run_random_agent(episodes=3, max_steps=100)
    evaluate_agent(q_table_random, ang_vel_bins_random, episodes=10, max_steps=100, agent_name="Aleatorio")

    # 2. Entrenamiento y evaluación del agente de Q-learning
    q_table_ql, ang_vel_bins_ql, rewards_ql = train_q_learning(episodes=10, max_steps=100)
    evaluate_agent(q_table_ql, ang_vel_bins_ql, episodes=10, max_steps=100, agent_name="Q-Learning")

    # 3. Entrenamiento y evaluación del agente de Monte Carlo
    q_table_mc, ang_vel_bins_mc, rewards_mc = train_monte_carlo(episodes=10, max_steps=100)
    evaluate_agent(q_table_mc, ang_vel_bins_mc, episodes=10, max_steps=100, agent_name="Monte Carlo")

    print("\n🎉 ¡Demo completada!")
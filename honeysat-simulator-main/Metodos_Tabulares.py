
from time import time
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

def softmax(x, tau=1.0):
    x = np.array(x)
    x = x - np.max(x)  # para estabilidad numérica
    exp_x = np.exp(x / tau)
    return exp_x / np.sum(exp_x)

def train_q_learning(episodes=500, max_steps=1000, learning_rate=0.5, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05):
    """Entrena un agente de Q-learning en el entorno de detumbling del CubeSat."""
    print("🚀 INICIANDO ENTRENAMIENTO DE Q-LEARNING")
    print(f"📈 Episodios totales: {episodes:,}")
    print("=" * 50)

    env = CubeSatDetumblingEnv(max_steps=max_steps)
    numero_bins =12
    # Bins de discretización para la velocidad angular (3 dimensiones)
    bins = np.linspace(-1.5, 1.5, numero_bins)
    ang_vel_bins = [bins, bins, bins]
    
    num_states = numero_bins**3
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    total_rewards = []
    
    rewards_per_episode = []
    steps_per_episode = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            # Eleccion Epsilon-greedy, sacando la opcion max

            if np.random.rand() < epsilon:
                # Acción aleatoria distinta de la acción greedy
                greedy_action = np.argmax(q_table[state, :])
                possible_actions = list(range(env.action_space.n))
                possible_actions.remove(greedy_action)
                action = np.random.choice(possible_actions)
            else:
                # Acción greedy
                action = np.argmax(q_table[state, :])
            # Utilizando softmax para selección de acción
            #probs = softmax(q_table[state, :], tau=1.0)
            #action = np.random.choice(np.arange(num_actions), p=probs)
            
            # Imprimir velocidad angular antes de aplicar la acción
            #print(f"Step: {step_count}")
            #print("Velocidad angular normal antes: {:.4f} rad/s".format(np.linalg.norm(obs[4:7])))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Imprimir velocidad angular después de aplicar la acción
            #print("Velocidad angular normal despues: {:.4f} rad/s".format(np.linalg.norm(next_obs[4:7])))
            #print("-" * 50)
            next_state = discretize_state(next_obs, ang_vel_bins)
            done = terminated or truncated
            total_reward += reward
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state
            obs = next_obs
            step_count += 1
            

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(total_reward)

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)
        
        print(f"Episodio {episode+1}/{episodes}: Recompensa total = {total_reward:.2f}, Pasos = {step_count}, Velocidad angular normal = {np.linalg.norm(obs[4:7]):.2f}")

    env.close()
    
    # Save the Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    plot_results(rewards_per_episode, steps_per_episode, episodes)

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

def running_average(x, N):
    """
    Calcula el promedio móvil (running average) de una lista o array.
    Se utiliza para suavizar las curvas de rendimiento.
    """
    import numpy as np
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0))
    # Utiliza la diferenciación de sumas acumuladas para eficiencia
    return (cumsum[N:] - cumsum[:-N]) / N

def plot_results(rewards, steps, episodes):
    """
    Crea los gráficos de rendimiento: Retorno por episodio y Pasos por episodio.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = 100 # Ventana de suavizado: 100 episodios
    
    # 1. Gráfico de Retorno/Recompensa por Episodio (Suavizado)
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, label='Retorno por Episodio', alpha=0.3)

    if len(rewards) >= N:
        smoothed_rewards = running_average(rewards, N)
        # Se plotea a partir del episodio N
        plt.plot(np.arange(N, episodes + 1), smoothed_rewards, color='red', label=f'Promedio Móvil (N={N})')

    plt.title('Rendimiento del Agente (Q-Learning) - Retorno por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Retorno Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 2. Gráfico de Pasos por Episodio
    plt.figure(figsize=(12, 5))
    plt.plot(steps, label='Pasos por Episodio', alpha=0.3)
    
    if len(steps) >= N:
        smoothed_steps = running_average(steps, N)
        plt.plot(np.arange(N, episodes + 1), smoothed_steps, color='red', label=f'Promedio Móvil (N={N})')

    plt.title('Rendimiento del Agente (Q-Learning) - Pasos por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos tomados')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

def test_q_learning(q_table, ang_vel_bins, episodes=100, max_steps=100, agent_name="Q-Learning"):
    """
    Evalúa el rendimiento de un agente Q-Learning entrenado (sin exploración y sin aprendizaje).
    No modifica la Q-table. Es el método de prueba para comparar hiperparámetros.
    """
    from cubesat_detumbling_rl import CubeSatDetumblingEnv
    import numpy as np

    print("\n" + "=" * 50)
    print(f"🧪 INICIANDO PRUEBA DEL AGENTE {agent_name.upper()}")
    print(f"🔬 Episodios de prueba: {episodes}")
    print("=" * 50)

    eval_env = CubeSatDetumblingEnv(max_steps=max_steps, granularity=10)
    total_rewards = []
    total_steps = []
    success_count = 0

    for episode in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < max_steps:
            state = discretize_state(obs, ang_vel_bins)
            # Elección de acción puramente greedy (explotación)
            action = np.argmax(q_table[state, :]) 
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        
        total_rewards.append(total_reward)
        total_steps.append(step_count)

        if terminated:
            success_count += 1
            print(f"  ✅ ÉXITO! Detumbling logrado en {step_count} pasos. Recompensa: {total_reward:.2f}")
        else:
            print(f"  ❌ Episodio finalizado después de {step_count} pasos. Recompensa: {total_reward:.2f}")

    eval_env.close()

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print("\n" + "=" * 50)
    print(f"📊 RESUMEN DE LA PRUEBA ({agent_name.upper()})")
    print(f"🏆 Tasa de éxito: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"📈 Recompensa promedio: {avg_reward:.2f}")
    print(f"🚶 Pasos promedio: {avg_steps:.2f}")
    print("=" * 50)
    return avg_reward, avg_steps

if __name__ == "__main__":
    print("🛰️  DEMO DE AGENTE DE APRENDIZAJE POR REFUERZO DE DETUMBLING")
    print()
    
    # 1. Evaluación del agente aleatorio (línea de base)
    #q_table_random, ang_vel_bins_random, rewards_random = run_random_agent(episodes=3, max_steps=100)
    #evaluate_agent(q_table_random, ang_vel_bins_random, episodes=10, max_steps=100, agent_name="Aleatorio")

    # 2. Entrenamiento y evaluación del agente de Q-learning
    q_table_ql, ang_vel_bins_ql, rewards_ql = train_q_learning(episodes=500, max_steps=1000)
    

    # 3. Entrenamiento y evaluación del agente de Monte Carlo
    #q_table_mc, ang_vel_bins_mc, rewards_mc = train_monte_carlo(episodes=10, max_steps=100)
    #evaluate_agent(q_table_mc, ang_vel_bins_mc, episodes=10, max_steps=100, agent_name="Monte Carlo")

    print("\n🎉 ¡Demo completada!")
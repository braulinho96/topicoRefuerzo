import time as timer
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

def plot_random_agent_results(mean_rewards, std_rewards, mean_steps, std_steps, episodes):
    """
    Gráficos del Agente Aleatorio:
    - Línea de promedio por episodio
    - Error por episodio (± std) con error bars
    - Línea de tendencia (promedio móvil con running_average)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N = 100  # ventana de suavizado
    x = np.arange(1, episodes + 1)

    # ============================
    # 1) RECOMPENSAS
    # ============================
    plt.figure(figsize=(12, 5))

    # Línea del retorno promedio por episodio
    plt.plot(
        x,
        mean_rewards,
        color='navy',
        linewidth=1.2,
        alpha=0.9,
        label='Retorno promedio por episodio'
    )

    # Barras de error (± std) SIN línea adicional
    plt.errorbar(
        x,
        mean_rewards,
        yerr=std_rewards,
        fmt='none',            # sin línea ni marcador extra
        ecolor='deepskyblue',  # color intenso para el error
        elinewidth=1,
        capsize=2,
        alpha=0.8,
        label='± 1 std'
    )

    # Línea de tendencia (running average)
    if len(mean_rewards) >= N:
        smoothed_rewards = running_average(mean_rewards, N)
        x_smooth = np.arange(N, episodes + 1)
        plt.plot(
            x_smooth,
            smoothed_rewards,
            color='crimson',
            linewidth=2,
            label=f'Promedio móvil (N={N})'
        )

    plt.title('Agente ALEATORIO - Retorno promedio por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Retorno total promedio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('random_agent_rewards_performance.png')

    # ============================
    # 2) PASOS
    # ============================
    plt.figure(figsize=(12, 5))

    # Línea de pasos promedio
    plt.plot(
        x,
        mean_steps,
        color='darkgreen',
        linewidth=1.2,
        alpha=0.9,
        label='Pasos promedio por episodio'
    )

    # Barras de error de pasos
    plt.errorbar(
        x,
        mean_steps,
        yerr=std_steps,
        fmt='none',
        ecolor='limegreen',
        elinewidth=1,
        capsize=2,
        alpha=0.8,
        label='± 1 std'
    )

    # Running average de pasos
    if len(mean_steps) >= N:
        smoothed_steps = running_average(mean_steps, N)
        x_smooth = np.arange(N, episodes + 1)
        plt.plot(
            x_smooth,
            smoothed_steps,
            color='crimson',
            linewidth=2,
            label=f'Promedio móvil (N={N})'
        )

    plt.title('Agente ALEATORIO - Pasos promedio por episodio (5 corridas)')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos promedio tomados')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('random_agent_steps_performance.png')

    plt.show()

def run_random_agent(episodes=100, max_steps=3000, repeats_per_step=5):
    print("=" * 50)
    print("🎲 INICIANDO EVALUACIÓN DEL AGENTE ALEATORIO (REPETICIONES POR PASO)")
    print(f"📈 Episodios:          {episodes}")
    print(f"⏱ Máx. pasos/episodio: {max_steps}")
    print(f"🔁 Repeticiones/paso:  {repeats_per_step}")
    print("=" * 50)

    env = CubeSatDetumblingEnv(render_mode=None, max_steps=max_steps, granularity=10)

    mean_rewards_per_episode = []
    std_rewards_per_episode = []
    mean_steps_per_episode = []
    std_steps_per_episode = []  # no estimamos error en pasos, lo dejamos en 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        episode_return_mean = 0.0          # suma de recompensas medias por paso
        step_index = 0                     # cuenta de pasos efectivos del episodio
        step_std_list = []                 # guardamos std de cada paso para propagar error

        while not done and step_index < max_steps:

            rewards_this_step = []

            for k in range(repeats_per_step):
                if done or step_index >= max_steps:
                    break

                # Elegir una accion aleatoria y ejecutarla
                action = env.action_space.sample()

                next_obs, reward, terminated, truncated, _ = env.step(action)

                rewards_this_step.append(reward)
                obs = next_obs
                done = terminated or truncated
                step_index += 1

                if done or step_index >= max_steps:
                    break

            if len(rewards_this_step) == 0:
                # El episodio terminó justo antes de poder ejecutar este paso "macro"
                break

            # 3) Estadísticas del paso
            mean_step_reward = float(np.mean(rewards_this_step))
            std_step_reward = float(np.std(rewards_this_step))

            episode_return_mean += mean_step_reward
            step_std_list.append(std_step_reward)

        # 4) Error del episodio: propagamos std de los pasos
        #    Asumimos independencia y sumamos varianzas: Var(total) ≈ Σ(std_step^2)
        if len(step_std_list) > 0:
            episode_std = float(np.sqrt(np.sum(np.array(step_std_list) ** 2)))
        else:
            episode_std = 0.0

        mean_rewards_per_episode.append(episode_return_mean)
        std_rewards_per_episode.append(episode_std)
        mean_steps_per_episode.append(step_index)
        std_steps_per_episode.append(0.0)  # no estimamos error de pasos por episodio

        print(
            f"Episodio {ep+1}/{episodes}: "
            f"Retorno medio (suma medias pasos) = {episode_return_mean:.2f}, "
            f"Error episodio ≈ {episode_std:.2f}, "
            f"Pasos efectivos = {step_index}"
        )

    env.close()

    # Convertir a arrays por si los necesitas después
    mean_rewards_per_episode = np.array(mean_rewards_per_episode)
    std_rewards_per_episode = np.array(std_rewards_per_episode)
    mean_steps_per_episode = np.array(mean_steps_per_episode)
    std_steps_per_episode = np.array(std_steps_per_episode)

    # Graficar estilo Q-Learning (línea + running average + error)
    plot_random_agent_results(
        mean_rewards_per_episode,
        std_rewards_per_episode,
        mean_steps_per_episode,
        std_steps_per_episode,
        episodes
    )

    return mean_rewards_per_episode, std_rewards_per_episode

def softmax(x, tau=1.0):
    x = np.array(x)
    x = x - np.max(x)  # para estabilidad numérica
    exp_x = np.exp(x / tau)
    return exp_x / np.sum(exp_x)

def train_q_learning(episodes=2000, max_steps=500, learning_rate=0.5, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05, fast_mode=True, granularity=5):
    """
    Entrena un agente de Q-learning en el entorno de detumbling del CubeSat.
    
    Args:
        episodes: Número de episodios de entrenamiento
        max_steps: Máximo número de pasos por episodio
        learning_rate: Tasa de aprendizaje (alpha)
        discount_factor: Factor de descuento (gamma)
        epsilon: Probabilidad inicial de exploración
        epsilon_decay: Factor de decaimiento de epsilon
        min_epsilon: Mínimo valor de epsilon
        fast_mode: Desactiva cálculos magnéticos/orbitales para acelerar ~3-5x
        granularity: Granularidad de la simulación (más bajo = más rápido)
    """
    
    
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO DE Q-LEARNING")
    print("="*60)
    print(f"📈 Episodios totales:      {episodes:,}")
    print(f"⏱ Máx. pasos/episodio:     {max_steps}")
    print(f"📊 Learning rate (α):       {learning_rate}")
    print(f"💰 Discount factor (γ):     {discount_factor}")
    print(f"🎲 Epsilon inicial:         {epsilon}")
    print(f"⚡ Optimizaciones:")
    print(f"   - fast_mode: {fast_mode}")
    print(f"   - granularity: {granularity}")
    print("="*60 + "\n")

    start_time = timer.time()
    
    env = CubeSatDetumblingEnv(
        max_steps=max_steps,
        granularity=granularity,
        render_mode=None,
        fast_mode=fast_mode
    )
    numero_bins = 12
    # Bins de discretización para la velocidad angular (3 dimensiones)
    bins = np.linspace(-1.5, 1.5, numero_bins)
    ang_vel_bins = [bins, bins, bins]
    
    num_states = numero_bins**3
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    total_rewards = []
    
    rewards_per_episode = []
    steps_per_episode = []
    success_per_episode = []
    
    total_successes = 0
    
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

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, ang_vel_bins)
            done = terminated or truncated
            total_reward += reward
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state
            obs = next_obs
            step_count += 1
            
            if terminated:
                total_successes += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(total_reward)

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)
        success_per_episode.append(1 if terminated else 0)
        
        # Imprimir progreso
        if (episode + 1) % max(1, episodes // 10) == 0:
            elapsed = timer.time() - start_time
            avg_reward = np.mean(rewards_per_episode[-max(100, episodes//10):])
            avg_steps = np.mean(steps_per_episode[-max(100, episodes//10):])
            success_rate = np.mean(success_per_episode[-max(100, episodes//10):]) * 100
            eps_per_sec = (episode + 1) / elapsed
            
            print(f"Episodio {episode+1}/{episodes} | "
                  f"Recompensa: {avg_reward:.2f} | "
                  f"Pasos: {avg_steps:.1f} | "
                  f"Éxito: {success_rate:.1f}% | "
                  f"ε: {epsilon:.4f} | "
                  f"Velocidad: {eps_per_sec:.2f} ep/s")

    env.close()
    
    elapsed_total = timer.time() - start_time
    
    # Resumen final
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"⏱️  Tiempo total: {elapsed_total/60:.1f} minutos ({elapsed_total:.0f}s)")
    print(f"📊 Promedio: {elapsed_total/episodes*1000:.1f}ms por episodio")
    print(f"🏆 Tasa de éxito total: {total_successes}/{episodes} ({total_successes/episodes*100:.1f}%)")
    print(f"📈 Recompensa promedio final (últimos 100): {np.mean(rewards_per_episode[-100:]):.2f}")
    print(f"🚶 Pasos promedio final (últimos 100): {np.mean(steps_per_episode[-100:]):.2f}")
    print("="*60 + "\n")
    
    # Save the Q-table
    with open("q_table_qlearning.pkl", "wb") as f:
        pickle.dump(q_table, f)

    plot_results(rewards_per_episode, steps_per_episode, episodes)

    return q_table, ang_vel_bins, total_rewards

def running_average(x, N):
    """
    Calcula el promedio móvil (running average) de una lista o array.
    Se utiliza para suavizar las curvas de rendimiento.
    """
    import numpy as np
    cumsum = np.cumsum(np.insert(np.array(x), 0, 0))
    # Utiliza la diferenciación de sumas acumuladas para eficiencia
    return (cumsum[N:] - cumsum[:-N]) / N

def plot_results(rewards, steps, episodes, agent_name="Q-Learning"):
    """
    Crea los gráficos de rendimiento: Retorno por episodio y Pasos por episodio.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = max(50, episodes // 20)  # Ventana de suavizado adaptativa
    
    # 1. Gráfico de Retorno/Recompensa por Episodio (Suavizado)
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, label='Retorno por Episodio', alpha=0.3)

    if len(rewards) >= N:
        smoothed_rewards = running_average(rewards, N)
        # Se plotea a partir del episodio N
        plt.plot(np.arange(N, episodes + 1), smoothed_rewards, color='red', label=f'Promedio Móvil (N={N})')

    plt.title(f'Rendimiento del Agente ({agent_name}) - Retorno por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Retorno Total')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'q_learning_performance_rewards_{agent_name.lower().replace("-", "_")}.png')
    
    # 2. Gráfico de Pasos por Episodio
    plt.figure(figsize=(12, 5))
    plt.plot(steps, label='Pasos por Episodio', alpha=0.3)
    
    if len(steps) >= N:
        smoothed_steps = running_average(steps, N)
        plt.plot(np.arange(N, episodes + 1), smoothed_steps, color='red', label=f'Promedio Móvil (N={N})')

    plt.title(f'Rendimiento del Agente ({agent_name}) - Pasos por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Pasos tomados')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'q_learning_performance_{agent_name.lower().replace("-", "_")}.png')
    
    plt.show()

def train_sarsa_td(episodes=1000, max_steps=200, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, fast_mode=True, granularity=5):
    """
    Entrena un agente de SARSA (Temporal Difference) para el problema de detumbling.
    SARSA es un algoritmo on-policy que actualiza basándose en la acción que realmente se toma.
    
    Args:
        episodes: Número de episodios de entrenamiento
        max_steps: Máximo número de pasos por episodio
        learning_rate: Tasa de aprendizaje (alpha)
        discount_factor: Factor de descuento (gamma)
        epsilon: Probabilidad inicial de exploración
        epsilon_decay: Factor de decaimiento de epsilon
        min_epsilon: Mínimo valor de epsilon
        fast_mode: Desactiva cálculos magnéticos/orbitales para acelerar ~3-5x
        granularity: Granularidad de la simulación (más bajo = más rápido). Por defecto 5 en entrenamiento
    
    Returns:
        q_table: Tabla Q entrenada
        ang_vel_bins: Bins de discretización para velocidad angular
        rewards_per_episode: Lista de recompensas por episodio
    """
    print("\n" + "=" * 60)
    print("🚀 INICIANDO ENTRENAMIENTO DE AGENTE SARSA (TEMPORAL DIFFERENCE)")
    print(f"📈 Episodios: {episodes:,}")
    print(f"⏱ Máx. pasos/episodio: {max_steps}")
    print(f"📊 Learning rate (α): {learning_rate}")
    print(f"💰 Discount factor (γ): {discount_factor}")
    print(f"🎲 Epsilon inicial: {epsilon}")
    print("=" * 60)

    env = CubeSatDetumblingEnv(max_steps=max_steps, granularity=granularity, render_mode=None, fast_mode=fast_mode)
    
    # Configuración de bins de discretización (más fina para mejor aprendizaje)
    numero_bins = 15
    bins = np.linspace(-1.5, 1.5, numero_bins)
    ang_vel_bins = [bins, bins, bins]
    
    num_states = numero_bins ** 3
    num_actions = env.action_space.n
    
    # Inicializar tabla Q
    q_table = np.zeros((num_states, num_actions))
    
    # Listas para tracking del desempeño
    rewards_per_episode = []
    steps_per_episode = []
    success_per_episode = []
    
    # Variables para estadísticas
    total_successes = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)
        
        # Epsilon-greedy: seleccionar acción inicial
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        
        done = False
        total_reward = 0
        step_count = 0
        episode_success = False
        
        while not done and step_count < max_steps:
            # Ejecutar acción
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs, ang_vel_bins)
            
            # Seleccionar siguiente acción (epsilon-greedy)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state, :])
            
            # Actualización SARSA: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            # Diferencia clave con Q-Learning: usa Q(s',a') en lugar de max(Q(s',:))
            q_table[state, action] = (q_table[state, action] + 
                                      learning_rate * (reward + 
                                                      discount_factor * q_table[next_state, next_action] - 
                                                      q_table[state, action]))
            
            # Actualizar estado y acción
            state = next_state
            action = next_action
            total_reward += reward
            step_count += 1
            
            done = terminated or truncated
            if terminated:
                episode_success = True
        
        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Tracking de desempeño
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)
        success_per_episode.append(1 if episode_success else 0)
        
        if episode_success:
            total_successes += 1
        
        # Imprimir progreso
        if (episode + 1) % max(1, episodes // 10) == 0:
            avg_reward = np.mean(rewards_per_episode[-max(100, episodes//10):])
            avg_steps = np.mean(steps_per_episode[-max(100, episodes//10):])
            success_rate = np.mean(success_per_episode[-max(100, episodes//10):]) * 100
            
            print(f"Episodio {episode+1}/{episodes} | "
                  f"Recompensa: {avg_reward:.2f} | "
                  f"Pasos: {avg_steps:.1f} | "
                  f"Éxito: {success_rate:.1f}% | "
                  f"ε: {epsilon:.4f}")
    
    env.close()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"🏆 Tasa de éxito total: {total_successes}/{episodes} ({total_successes/episodes*100:.1f}%)")
    print(f"📊 Recompensa promedio final (últimos 100): {np.mean(rewards_per_episode[-100:]):.2f}")
    print(f"🚶 Pasos promedio final (últimos 100): {np.mean(steps_per_episode[-100:]):.2f}")
    print("=" * 60 + "\n")
    
    # Guardar tabla Q
    with open("q_table_sarsa_td.pkl", "wb") as f:
        pickle.dump(q_table, f)
    
    # Graficar resultados
    plot_td_results(rewards_per_episode, steps_per_episode, success_per_episode, episodes)
    
    return q_table, ang_vel_bins, rewards_per_episode

def plot_td_results(rewards, steps, successes, episodes):
    """
    Crea gráficos detallados del rendimiento del agente TD.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = max(50, episodes // 20)  # Ventana de suavizado adaptativa
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Rendimiento del Agente SARSA (TD)', fontsize=16)
    
    # 1. Recompensa por episodio
    axes[0, 0].plot(rewards, label='Recompensa por episodio', alpha=0.3, color='blue')
    if len(rewards) >= N:
        smoothed_rewards = running_average(rewards, N)
        x_smooth = np.arange(N, episodes + 1)
        axes[0, 0].plot(x_smooth, smoothed_rewards, color='darkblue', linewidth=2, label=f'Promedio móvil (N={N})')
    axes[0, 0].set_title('Recompensa Total por Episodio')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Recompensa')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    
    # 2. Pasos por episodio
    axes[0, 1].plot(steps, label='Pasos por episodio', alpha=0.3, color='green')
    if len(steps) >= N:
        smoothed_steps = running_average(steps, N)
        x_smooth = np.arange(N, episodes + 1)
        axes[0, 1].plot(x_smooth, smoothed_steps, color='darkgreen', linewidth=2, label=f'Promedio móvil (N={N})')
    axes[0, 1].set_title('Pasos por Episodio')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Número de pasos')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    
    # 3. Tasa de éxito acumulada
    cumsum_success = np.cumsum(successes)
    success_rate = cumsum_success / np.arange(1, episodes + 1)
    axes[1, 0].plot(success_rate, color='red', linewidth=1.5)
    if len(success_rate) >= N:
        smoothed_success = running_average(success_rate, N)
        x_smooth = np.arange(N, episodes + 1)
        axes[1, 0].plot(x_smooth, smoothed_success, color='darkred', linewidth=2, label=f'Promedio móvil (N={N})')
    axes[1, 0].set_title('Tasa de Éxito Acumulada')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Tasa de éxito')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    
    # 4. Distribución de pasos en últimos 200 episodios
    last_episodes = min(200, episodes)
    axes[1, 1].hist(steps[-last_episodes:], bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(steps[-last_episodes:]), color='red', linestyle='--', linewidth=2, label='Media')
    axes[1, 1].axvline(np.median(steps[-last_episodes:]), color='green', linestyle='--', linewidth=2, label='Mediana')
    axes[1, 1].set_title(f'Distribución de Pasos (últimos {last_episodes} episodios)')
    axes[1, 1].set_xlabel('Número de pasos')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('sarsa_td_training_results.png', dpi=100, bbox_inches='tight')
    print("📊 Gráficos guardados en 'sarsa_td_training_results.png'")
    plt.show()

if __name__ == "__main__":
    print("🛰️  DEMO DE AGENTE DE APRENDIZAJE POR REFUERZO DE DETUMBLING")
    print("="*60)
    print("Elige qué agente entrenar:\n")
    print("1. SARSA (Temporal Difference) - On-policy, convergencia más rápida")
    print("2. Q-Learning - Off-policy, convergencia garantizada\n")
    
    choice = input("Selecciona (1 o 2): ").strip()
    
    if choice == "1":
        # ========================================
        # ENTRENAR AGENTE SARSA (TEMPORAL DIFFERENCE)
        # ========================================
        print("\n" + "="*60)
        print("🚀 AGENTE SARSA (Temporal Difference)")
        print("="*60)
        print("⚙️ OPTIMIZACIONES ACTIVAS:")
        print("   ✓ fast_mode=True    (Desactiva magnético/orbital)")
        print("   ✓ granularity=5     (Menos iteraciones internas)")
        print("   ✓ epsilon_decay=0.98 (Exploración más rápida)")
        print("="*60 + "\n")
        
        start_time = timer.time()
        
        q_table_td, ang_vel_bins_td, rewards_td = train_sarsa_td(
            episodes=1000,
            max_steps=150,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.05,
            fast_mode=True,
            granularity=5
        )
        
        elapsed = timer.time() - start_time
        print(f"\n⏱️  Tiempo total: {elapsed/60:.1f} minutos")
        print(f"📊 Promedio: {elapsed/1000*1000:.1f}ms por episodio")
        
    elif choice == "2":
        # ========================================
        # ENTRENAR AGENTE Q-LEARNING
        # ========================================
        print("\n" + "="*60)
        print("🚀 AGENTE Q-LEARNING")
        print("="*60)
        print("⚙️ OPTIMIZACIONES ACTIVAS:")
        print("   ✓ fast_mode=True    (Desactiva magnético/orbital)")
        print("   ✓ granularity=5     (Menos iteraciones internas)")
        print("="*60 + "\n")
        
        start_time = timer.time()
        
        q_table_ql, ang_vel_bins_ql, rewards_ql = train_q_learning(
            episodes=100000,
            max_steps=200,
            learning_rate=0.5,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.99,
            min_epsilon=0.05,
            fast_mode=True,
            granularity=5
        )
        
        elapsed = timer.time() - start_time
        print(f"\n⏱️  Tiempo total: {elapsed/60:.1f} minutos")
        print(f"📊 Promedio: {elapsed/1000*1000:.1f}ms por episodio")
    
    else:
        print("❌ Opción inválida. Ejecuta de nuevo y selecciona 1 o 2.")
    
    print("\n🎉 ¡Entrenamiento completado!")
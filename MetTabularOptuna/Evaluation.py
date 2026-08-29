import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ============================
# CONFIGURACIÓN DE RUTAS Y Q-TABLE
# ============================
def load_action_map(config_path="best_config_sarsa.json"):
	with open(config_path, "r") as f:
		config = json.load(f)
	factors = config["action_map_info"]
	max_t = factors["max_torque_constant"]
	#min_t = factors["min_torque_factor"] * max_t
	#mid_t = factors["mid_torque_factor"] * max_t
	#micro_t = factors["micro_torque_factor"] * max_t

	
	action_map = {
		0: np.array([max_t, 0, 0]),  
		1: np.array([-max_t, 0, 0]),
		2: np.array([0, max_t, 0]),  
		3: np.array([0, -max_t, 0]),
		4: np.array([0, 0, max_t]),  
		5: np.array([0, 0, -max_t]),
		6: np.array([0, 0, 0]),
	}
	return action_map

def load_q_table(q_path):
	ext = os.path.splitext(q_path)[1].lower()
	if ext == ".npy":
		return np.load(q_path, allow_pickle=True)
	elif ext in (".pkl", ".pickle"):
		with open(q_path, "rb") as f:
			return pickle.load(f)
	elif ext == ".json":
		with open(q_path, "r") as f:
			return json.load(f)
	else:
		raise ValueError("Formato de Q-table no soportado. Usa .npy, .pkl o .json")

def get_env():
	from cubesat_detumbling_rl import CubeSatDetumblingEnv
	from SatellitePersonality import SatellitePersonality
	return CubeSatDetumblingEnv

def select_action(q_table, state, n_actions):
	if isinstance(q_table, np.ndarray):
		idx = int(state)
		return int(np.argmax(q_table[idx]))
	elif isinstance(q_table, dict):
		key = str(state)
		if key in q_table:
			return int(np.argmax(q_table[key]))
		return 0
	else:
		return 0

def evaluate_q_table(q_table, env_class, action_map, n_episodes=10000, max_steps=400, threshold=0.01):
		numero_bins = 15
		bins = np.linspace(-1.5, 1.5, numero_bins)
		ang_vel_bins = [bins, bins, bins]
		def discretize_state(observation, bins):
			discretized = []
			for i in range(4, 7):
				clipped_obs = np.clip(observation[i], bins[i-4][0], bins[i-4][-1])
				d = np.digitize(clipped_obs, bins[i-4]) - 1
				d = min(max(d, 0), len(bins[i-4]) - 1)
				discretized.append(d)
			return int(sum(d * (len(b) - 1)**i for i, (d, b) in enumerate(zip(discretized, bins))))

		success_count = 0
		all_angular_traces = []
		steps_list = []
		rewards_list = []
		n_actions = len(action_map)
		for ep in range(n_episodes):
			if (ep + 1) % 500 == 0 or ep == 0:
				print(f"Progreso: Episodio {ep + 1} de {n_episodes}")
			env = env_class(max_steps=max_steps, action_map=action_map)
			obs, _ = env.reset()
			done, truncated = False, False
			trace = [np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity)]
			step = 0
			total_reward = 0
			while not (done or truncated) and step < max_steps:
				state_idx = discretize_state(obs, ang_vel_bins)
				action_idx = select_action(q_table, state_idx, n_actions)
				obs, reward, done, truncated, _ = env.step(action_idx)
				total_reward += reward
				trace.append(np.linalg.norm(env.unwrapped.rotation_sim.angular_velocity))
				step += 1
			if trace[-1] < threshold:
				success_count += 1
			all_angular_traces.append(trace)
			steps_list.append(step)
			rewards_list.append(total_reward)
			env.close()
		mean_steps = np.mean(steps_list)
		mean_reward = np.mean(rewards_list)
		print(f"✅ Tasa de Éxito Final: {(success_count/n_episodes)*100:.2f}% ({success_count}/{n_episodes})")
		print(f"📊 Media de pasos por episodio: {mean_steps:.2f}")
		print(f"🏅 Media de recompensa por episodio: {mean_reward:.4f}")
		#Utilizar el grafico para ver que sucede con la velocidad si realmente se reduce
		plt.figure(figsize=(10, 5))
		plt.plot(all_angular_traces[0], label="Velocidad Angular (rad/s)", color="blue")
		plt.axhline(y=threshold, color="red", linestyle="--", label="Umbral Éxito")
		plt.title("Frenado del Satélite - Q-Table Evaluación")
		plt.xlabel("Pasos")
		plt.ylabel("||ω||")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.show()

if __name__ == "__main__":
	print("\n=== Evaluación de Q-Table SARSA RF ===\n")
	qtable_folder = "q_tables"  #Carpeta de los pkl
	abs_folder = os.path.join(os.path.dirname(__file__), qtable_folder)
	if not os.path.exists(abs_folder):
		print(f"La carpeta '{qtable_folder}' no existe. Créala y coloca tus Q-tables allí.")
		exit(1)
	files = [f for f in os.listdir(abs_folder) if f.endswith((".npy", ".pkl", ".pickle", ".json"))]
	if not files:
		print(f"No se encontraron archivos de Q-table en '{qtable_folder}'.")
		exit(1)
	print("Archivos de Q-table disponibles:")
	for i, fname in enumerate(files):
		print(f"  [{i+1}] {fname}")
	idx = int(input("Selecciona el número del archivo Q-table: ").strip()) - 1
	if idx < 0 or idx >= len(files):
		print("Selección inválida.")
		exit(1)
	q_path = os.path.join(abs_folder, files[idx])
	n_episodes = int(input("Número de episodios a evaluar: ").strip())
	max_steps = int(input("Máximo de pasos por episodio: ").strip())
	action_map = load_action_map("best_config_sarsa.json")
	q_table = load_q_table(q_path)
	env_class = get_env()
	evaluate_q_table(q_table, env_class, action_map, n_episodes=n_episodes, max_steps=max_steps)

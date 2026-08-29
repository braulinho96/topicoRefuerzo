import gymnasium as gym
import numpy as np
import pickle
import time

from cubesat_detumbling_rl import CubeSatDetumblingEnv


def discretize_state(observation, bins):
    """Discretize a continuous observation into a single integer."""
    discretized = []
    # Discretize angular velocity (observation[4:7])
    for i in range(4, 7):
        # Clip observation to be within the range of bins
        clipped_obs = np.clip(observation[i], bins[i-4][0], bins[i-4][-1])
        digit = np.digitize(clipped_obs, bins[i-4]) - 1
        discretized.append(digit)
    
    # Combine the discretized parts into a single state index
    # This is a simple way to create a unique index for each state combination
    # For 3 dimensions with 10 bins each, this gives 10*10*10 = 1000 states
    return sum(d * (len(b) - 1)**i for i, (d, b) in enumerate(zip(discretized, bins)))

def train_q_learning(episodes=10):
    """Train a Q-learning agent on the CubeSat Detumbling Environment."""
    print("=" * 50)
    print("🚀 STARTING CUBESAT DETUMBLING TRAINING WITH Q-LEARNING")
    print(f"📈 Total episodes: {episodes:,}")
    print("=" * 50)

    env = CubeSatDetumblingEnv()

    # Discretization bins for angular velocity (3 dimensions)
    # More bins = finer granularity, but larger Q-table
    n_bins = 10
    ang_vel_bins = [np.linspace(-1.5, 1.5, n_bins) for _ in range(3)]
    num_states = n_bins ** 3
    num_actions = env.action_space.n

    # Initialize Q-table with zeros
    q_table = np.zeros((num_states, num_actions))

    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay_rate = 0.999
    min_epsilon = 0.01

    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_state = discretize_state(new_obs, ang_vel_bins)
            done = terminated or truncated

            # Q-table update
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

            state = new_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

    # Save the Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    
    print("=" * 50)
    print("💾 Q-table saved to q_table.pkl")
    print("=" * 50)
    
    env.close()
    return q_table, ang_vel_bins

def evaluate_q_learning(q_table, ang_vel_bins, episodes=1):
    """Evaluate the trained Q-learning agent."""
    print("=" * 50)
    print("🧪 EVALUATING TRAINED Q-LEARNING AGENT")
    print("=" * 50)

    eval_env = CubeSatDetumblingEnv(render_mode='human', max_steps=100)
    success_count = 0
    total_rewards = []

    for episode in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\n🎮 Episode {episode + 1}/{episodes}")
        print("-" * 30)

        while not done and step_count < 100:
            state = discretize_state(obs, ang_vel_bins)
            action = np.argmax(q_table[state, :])
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            if step_count % 20 == 0:
                angular_vel_norm = np.linalg.norm(obs[4:7])
                print(f"  Step {step_count}: Angular velocity norm = {angular_vel_norm:.4f} rad/s")

        total_rewards.append(total_reward)

        if terminated:
            success_count += 1
            print(f"  ✅ SUCCESS! Detumbling achieved in {step_count} steps")
        else:
            print(f"  ⏰ Episode ended after {step_count} steps")

        print(f"  📊 Total Reward: {total_reward:.2f}")

    eval_env.close()

    print("\n" + "=" * 50)
    print("📈 EVALUATION SUMMARY")
    print(f"🏆 Success Rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"📊 Average Reward: {np.mean(total_rewards):.2f}")
    print("=" * 50)

if __name__ == "__main__":
    print("🛰️  CUBESAT DETUMBLING RL DEMO WITH Q-LEARNING")
    print()
    
    # Train the agent
    #q_table, bins = train_q_learning(episodes=200)

    # Evaluate the trained agent
    #evaluate_q_learning(q_table, bins, episodes=3)
    
    #print("\n🎉 Demo completed! Check the saved model: q_table.pkl")
    

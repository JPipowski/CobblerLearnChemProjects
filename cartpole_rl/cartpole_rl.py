import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize the environment (no human rendering during training for speed)
env = gym.make("CartPole-v1")

# 2. State Discretization Setup
# CartPole has continuous values, but tabular Q-learning requires discrete states.
# We divide each of the 4 state variables into bins.
num_bins = 6
bins = [
    np.linspace(-2.4, 2.4, num_bins - 1),  # Cart Position
    np.linspace(-3.0, 3.0, num_bins - 1),  # Cart Velocity
    np.linspace(-0.209, 0.209, num_bins - 1),  # Pole Angle (~12 degrees max)
    np.linspace(-3.0, 3.0, num_bins - 1)  # Pole Angular Velocity
]


def discretize_state(observation):
    """Converts continuous observation floats into discrete bin indices."""
    state_indices = []
    for i, val in enumerate(observation):
        idx = np.digitize(val, bins[i])
        state_indices.append(idx)
    return tuple(state_indices)


# 3. Q-Learning Hyperparameters
num_episodes = 600
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Reduces exploration over time
min_epsilon = 0.01

# Initialize Q-table: 6^4 states, 2 possible actions (left=0, right=1)
q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, env.action_space.n))

episode_rewards = []

print("--- Starting Q-Learning Training ---")

# 4. Training Loop
for episode in range(num_episodes):
    obs, info = env.reset()
    state = discretize_state(obs)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Epsilon-greedy action selection (Exploration vs. Exploitation)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random)
        else:
            action = np.argmax(q_table[state])  # Exploit (learned Q-values)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_obs)

        # Q-learning Update Equation
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_factor * q_table[next_state + (best_next_action,)] * (not terminated)
        td_error = td_target - q_table[state + (action,)]
        q_table[state + (action,)] += learning_rate * td_error

        state = next_state
        total_reward += reward

    # Decay epsilon after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    # Print progress every 50 episodes
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward} | Epsilon: {epsilon:.2f}")

env.close()
print("--- Training Completed ---")

# 5. Graphing the Learning Progress
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode Reward", color="dodgerblue", alpha=0.4)

# Calculate a moving average trendline to make learning growth clear
window = 20
if len(episode_rewards) >= window:
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"Moving Average ({window} eps)",
             color="crimson", linewidth=2.5)

plt.xlabel("Episode", fontsize=12)
plt.ylabel("Total Reward (Steps Survived)", fontsize=12)
plt.title("Tabular Q-Learning Training Curve on CartPole-v1", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
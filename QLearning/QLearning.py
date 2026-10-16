import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. ENVIRONMENT & HYPERPARAMETERS
# ==========================================
# Create environment with render_mode='ansi' to visualize navigation steps
env = gym.make("FrozenLake-v1", render_mode="ansi")

print("--- FrozenLake-v1 Environment Information ---")
print(f"Observation Space (States): {env.observation_space}")
print(f"Action Space (Actions):     {env.action_space}")

# Hyperparameter Experimentation Set
num_episodes = 5000
learning_rate = 0.1  # Alpha
discount_factor = 0.99  # Gamma
max_epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.001  # Epsilon decay rate per episode

# ==========================================
# 2. INITIALIZE Q-TABLE
# ==========================================
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

print(f"\n--- Q-Table Initialized ({num_states} states x {num_actions} actions) ---")

# ==========================================
# 3. TRAINING AGENT WITH EPSILON-GREEDY
# ==========================================
print("\n--- Training Agent ---")
rewards_all_episodes = []
epsilon = max_epsilon

for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_episode_reward = 0

    while not (terminated or truncated):
        # Exploration-exploitation trade-off (Epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore (random action)
        else:
            action = np.argmax(q_table[state])  # Exploit (best learned action)

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Bellman Equation Update
        current_q = q_table[state, action]
        max_future_q = np.max(q_table[next_state])
        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        q_table[state, action] = new_q

        state = next_state
        total_episode_reward += reward

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_all_episodes.append(total_episode_reward)

print("Training finished successfully!")

# ==========================================
# 4. SAVE Q-TABLE TO FILE
# ==========================================
q_table_filename = "frozenlake_q_table.npy"
np.save(q_table_filename, q_table)
print(f"\n[Success] Q-table saved to '{q_table_filename}' for later use.")

# (Optional Demonstration) Reloading Q-table from file:
# loaded_q_table = np.load(q_table_filename)


# ==========================================
# 5. GENERATE LEARNING CURVE PLOT
# ==========================================
# Calculate moving average of rewards over blocks of 100 episodes to show clear trends
block_size = 100
episodes_blocks = range(0, num_episodes, block_size)
moving_avg_rewards = [
    np.mean(rewards_all_episodes[i:i + block_size]) * 100  # Convert to success percentage per block
    for i in episodes_blocks
]

plt.figure(figsize=(9, 5))
plt.plot(episodes_blocks, moving_avg_rewards, color='blue', linewidth=2, marker='o')
plt.xlabel('Episode Blocks (x100)')
plt.ylabel('Success Rate (%)')
plt.title('Q-Learning Progress: Learning Curve (Moving Average)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("[Success] Learning curve plot saved as 'learning_curve.png'.")


# ==========================================
# 6. POLICY GRID DISPLAY & TESTING
# ==========================================
def display_policy_grid(q_tbl):
    arrow_mapping = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_grid = [arrow_mapping[np.argmax(q_tbl[state])] for state in range(16)]

    print("\n--- Learned Policy Grid (FrozenLake 4x4) ---")
    for r in range(4):
        row_str = " ".join(policy_grid[r * 4: (r + 1) * 4])
        print(f"[ {row_str} ]")


display_policy_grid(q_table)

# Test agent over 100 episodes (No exploration)
num_test_episodes = 100
success_count = 0

for _ in range(num_test_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        if terminated and reward == 1.0:
            success_count += 1

success_rate = (success_count / num_test_episodes) * 100
print(f"\nTest Success Rate ({num_test_episodes} Episodes): {success_rate:.1f}%")

# ==========================================
# 7. VISUALIZE AGENT NAVIGATING ENVIRONMENT
# ==========================================
print("\n--- Visualizing Agent Navigating the Environment (Test Run) ---")
test_env = gym.make("FrozenLake-v1", render_mode="ansi")
state, info = test_env.reset()
terminated = False
truncated = False

print(test_env.render())

step_count = 0
while not (terminated or truncated):
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = test_env.step(action)
    step_count += 1
    print(f"--- Step {step_count} (Action: {['LEFT', 'DOWN', 'RIGHT', 'UP'][action]}) ---")
    print(test_env.render())

    if terminated:
        if reward == 1.0:
            print("Goal reached successfully!")
        else:
            print("Agent fell into a hole!")

test_env.close()
env.close()
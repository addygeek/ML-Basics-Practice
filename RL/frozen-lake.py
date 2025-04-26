import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Environment details
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table initialization
q_table = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.8         # Learning rate
gamma = 0.95        # Discount factor
epsilon = 1.0       # Exploration probability
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 2000
max_steps = 100

# Mapping actions to directions for clarity
action_meaning = {
    0: '←',
    1: '↓',
    2: '→',
    3: '↑'
}

# Training loop
rewards_per_episode = []

for ep in range(episodes):
    state, _ = env.reset()
    total_rewards = 0
    done = False

    for step in range(max_steps):
        # ε-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        # Q-learning update rule
        q_old = q_table[state, action]
        q_max_next = np.max(q_table[next_state])
        q_table[state, action] = q_old + alpha * (reward + gamma * q_max_next - q_old)

        state = next_state
        total_rewards += reward

        if done:
            break

    rewards_per_episode.append(total_rewards)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("✅ Training Complete!")

# ---------------------------------------
# 🔍 Visualization of Q-table as Heatmaps
# ---------------------------------------
def plot_q_table(q_table, title="Q-table Heatmap"):
    plt.figure(figsize=(10, 6))
    for action in range(n_actions):
        plt.subplot(1, 4, action + 1)
        sns.heatmap(q_table[:, action].reshape(4, 4), annot=True, cbar=False,
                    cmap="YlGnBu", fmt=".2f")
        plt.title(f"Action: {action_meaning[action]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_q_table(q_table)

# ------------------------------------------------
# 🧠 Evaluation (Greedy policy)
# ------------------------------------------------
success_count = 0
episodes_test = 100

for ep in range(episodes_test):
    state, _ = env.reset()
    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        if done:
            if reward == 1:
                success_count += 1
            break

print(f"\n🎯 Success Rate after Training: {success_count}% over {episodes_test} episodes")

# ------------------------------------------------
# 🎮 Render 1 Playthrough
# ------------------------------------------------
state, _ = env.reset()
env.render()
print("🕹 Path:")
for step in range(max_steps):
    action = np.argmax(q_table[state])
    print(f"State: {state}, Action: {action_meaning[action]}")
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    env.render()
    if done:
        if reward == 1:
            print("🏁 Reached Goal!")
        else:
            print("💀 Fell into a Hole!")
        break

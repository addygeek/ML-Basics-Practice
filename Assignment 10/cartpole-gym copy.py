# Install dependencies (no Box2D needed for CartPole)
#pip install gymnasium tensorflow matplotlib numpy

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import time

# Set random seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset(seed=seed)

# Define the Actor-Critic model
class ActorCriticModel(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.common = layers.Dense(128, activation="relu")
        self.actor = layers.Dense(num_actions, activation="softmax")
        self.critic = layers.Dense(1)
    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

# Hyperparameters
num_actions = env.action_space.n
optimizer = keras.optimizers.Adam(learning_rate=0.01)
gamma = 0.99  # Discount factor
max_episodes = 500
max_steps_per_episode = 200
mse = keras.losses.MeanSquaredError()

model = ActorCriticModel(num_actions)
eps = np.finfo(np.float32).eps.item()

def train_actor_critic():
    episode_rewards = []
    moving_average_rewards = []
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        episode_reward = 0
        action_probs_history = []
        critic_value_history = []
        rewards_history = []
        with tf.GradientTape() as tape:
            for timestep in range(max_steps_per_episode):
                state_in = tf.expand_dims(state, 0)
                action_probs, critic_value = model(state_in)
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                action_probs_history.append(tf.math.log(action_probs[0, action]))
                critic_value_history.append(critic_value[0, 0])
                rewards_history.append(reward)
                state = tf.convert_to_tensor(next_state, dtype=tf.float32)
                if done:
                    break
            # Calculate expected returns
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            # Calculate losses
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
                advantage = ret - value
                actor_losses.append(-log_prob * advantage)
                critic_losses.append(mse([ret], [value]))
            loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        episode_rewards.append(episode_reward)
        if len(episode_rewards) >= 100:
            moving_avg = np.mean(episode_rewards[-100:])
            moving_average_rewards.append(moving_avg)
            print(f"Episode {episode}: Reward = {episode_reward}, Moving Average (100) = {moving_avg:.2f}")
            if moving_avg >= 195.0:
                print(f"Solved at episode {episode}!")
                break
        else:
            print(f"Episode {episode}: Reward = {episode_reward}")
    return episode_rewards, moving_average_rewards

print("Starting training...")
start_time = time.time()
rewards, moving_avg_rewards = train_actor_critic()
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Plot training progress
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.subplot(1, 2, 2)
plt.plot(moving_avg_rewards)
plt.axhline(y=195, color='r', linestyle='--', label='Solved Threshold')
plt.title('Moving Average Rewards (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.tight_layout()
plt.show()

# Visualization
def create_animation():
    state, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps_per_episode):
        frames.append(env.render())
        state_in = tf.convert_to_tensor(state, dtype=tf.float32)
        state_in = tf.expand_dims(state_in, 0)
        action_probs, _ = model(state_in)
        action = np.argmax(np.squeeze(action_probs))
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated:
            break
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.close()
    def animate(i):
        ax.clear()
        ax.imshow(frames[i])
        ax.set_axis_off()
        ax.set_title(f'Step: {i}')
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    return anim

print("Creating animation...")
anim = create_animation()
HTML(anim.to_jshtml())

env.close()

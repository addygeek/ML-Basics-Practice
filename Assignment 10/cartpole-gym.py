# First, make sure you have TensorFlow installed
# Run this in your terminal/command prompt:
# pip install tensorflow

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from IPython.display import HTML

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

# Initialize the model and training variables
model = ActorCriticModel(num_actions)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# Training function
def train_actor_critic():
    # Lists to track progress
    episode_rewards = []
    moving_average_rewards = []
    
    with tf.GradientTape() as tape:
        for episode in range(max_episodes):
            state, _ = env.reset()
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            episode_reward = 0
            
            # Storage for actions and rewards
            action_probs_history = []
            critic_value_history = []
            rewards_history = []
            
            for timestep in range(max_steps_per_episode):
                # Forward pass through the model
                state = tf.expand_dims(state, 0)
                action_probs, critic_value = model(state)
                
                # Sample action from action probability distribution
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                
                # Apply action to the environment
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                
                # Save actions and rewards
                action_probs_history.append(tf.math.log(action_probs[0, action]))
                critic_value_history.append(critic_value[0, 0])
                rewards_history.append(reward)
                
                state = tf.convert_to_tensor(next_state, dtype=tf.float32)
                
                if done or truncated:
                    break
            
            # Update running statistics
            episode_rewards.append(episode_reward)
            if len(episode_rewards) >= 100:
                moving_avg = np.mean(episode_rewards[-100:])
                moving_average_rewards.append(moving_avg)
                print(f"Episode {episode}: Reward = {episode_reward}, Moving Average (100) = {moving_avg:.2f}")
                
                # Early stopping if solved
                if moving_avg >= 195.0:
                    print(f"Solved at episode {episode}!")
                    break
            else:
                print(f"Episode {episode}: Reward = {episode_reward}")
            
            # Calculate expected returns
            returns = []
            discounted_sum = 0
            
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            
            # Normalize returns
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()
            
            # Calculate loss values
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            
            for log_prob, value, ret in history:
                # Actor loss
                advantage = ret - value
                actor_losses.append(-log_prob * advantage)
                
                # Critic loss
                critic_losses.append(keras.losses.mean_squared_error(tf.expand_dims(value, 0), 
                                                                     tf.expand_dims(ret, 0)))
            
            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Reset tape for next episode
            tape.reset()
    
    return episode_rewards, moving_average_rewards

# Train the model
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

# Function to create animation
def create_animation():
    # Reset environment
    state, _ = env.reset()
    frames = []
    
    for _ in range(max_steps_per_episode):
        # Render and capture frame
        frames.append(env.render())
        
        # Get action from model
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs, _ = model(state_tensor)
        action = np.argmax(np.squeeze(action_probs))
        
        # Apply action
        state, _, done, truncated, _ = env.step(action)
        
        if done or truncated:
            break
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.close()
    
    def animate(i):
        ax.clear()
        ax.imshow(frames[i])
        ax.set_axis_off()
        ax.set_title(f'Step: {i}')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    return anim

# Create and display animation
print("Creating animation...")
anim = create_animation()
HTML(anim.to_jshtml())

# Close the environment
env.close()

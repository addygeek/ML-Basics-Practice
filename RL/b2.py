import numpy as np
import gym
import random

def train_taxi_agent():
    # Create environment
    env = gym.make('Taxi-v3')
    
    # Initialize Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    learning_rate = 0.8
    discount_rate = 0.9
    epsilon = 1.0
    decay_rate = 0.005
    num_episodes = 2000
    max_steps = 99  # Per episode

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        for step in range(max_steps):
            # Exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(qtable[state, :])  # Exploit
            
            # Take action
            new_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )
            
            state = new_state
            if done:
                break
        
        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)
    
    print("Training completed")
    return env, qtable

def test_agent(env, qtable, max_steps=99):
    state = env.reset()
    done = False
    total_reward = 0
    
    print("Testing trained agent")
    for step in range(max_steps):
        action = np.argmax(qtable[state, :])
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        state = new_state
        if done:
            break
    
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    env, qtable = train_taxi_agent()
    test_agent(env, qtable)

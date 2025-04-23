import random
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation

class Environment:
    def __init__(self):
        self.states = [0, 1, 2]
        self.actions = [0, 1]  # 0 = left, 1 = right
        self.rewards = {2: 1}
        self.state = 0
        self.total_reward = 0
        self.step_count = 0
        
        # Setup the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.suptitle('Reinforcement Learning Visualization')
        
        # Initialize reward history
        self.reward_history = []
        self.reward_line, = self.ax2.plot([], [])
        
        # Setup the axes
        self.ax1.set_xlim(-0.5, 2.5)
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Reward')
        
    def step(self, action):
        if action == 0:  # left
            new_state = max(0, self.state - 1)
        else:  # right
            new_state = min(2, self.state + 1)
        reward = self.rewards.get(new_state, 0)
        self.state = new_state
        return new_state, reward
    
    def update(self, frame):
        self.ax1.clear()
        self.step_count += 1
        
        # Choose and perform action
        action = random.choice(self.actions)
        _, reward = self.step(action)
        self.total_reward += reward
        self.reward_history.append(reward)
        
        # Draw states
        for s in self.states:
            color = 'lightblue' if s != self.state else 'lightgreen'
            self.ax1.plot(s, 0, 'o', markersize=30, color=color)
            self.ax1.text(s, 0, f'S{s}', horizontalalignment='center', verticalalignment='center')
        
        # Draw action arrow
        direction = 1 if action == 1 else -1
        self.ax1.arrow(self.state, 0.1, direction * 0.4, 0,
                      head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Update plot properties
        self.ax1.set_xlim(-0.5, 2.5)
        self.ax1.set_ylim(-0.5, 0.5)
        self.ax1.set_title(f'Step: {self.step_count} | State: S{self.state} | Action: {"Right" if action == 1 else "Left"} | Total Reward: {self.total_reward}')
        self.ax1.axis('off')
        
        # Update reward history plot
        self.ax2.clear()
        self.ax2.plot(self.reward_history, 'b-')
        self.ax2.set_xlim(0, max(100, len(self.reward_history)))
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Reward')
        self.ax2.grid(True)
        
        plt.tight_layout()
        return self.ax1, self.ax2

def run_continuous_simulation():
    env = Environment()
    anim = FuncAnimation(env.fig, env.update, frames=None, 
                        interval=500, blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    run_continuous_simulation()

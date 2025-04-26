import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import random
from typing import Tuple, Dict, List
import time
from dataclasses import dataclass
from enum import Enum
import json

class MazeElement(Enum):
    EMPTY = 0
    WALL = 1
    GOAL = 2
    START = 3
    TRAP = 4
    REWARD = 5

@dataclass
class TrainingConfig:
    alpha: float = 0.1          # Learning rate
    gamma: float = 0.9          # Discount factor
    epsilon: float = 1.0        # Starting exploration rate (changed from 0.1 to 1.0)
    episodes: int = 500         # Number of training episodes
    min_epsilon: float = 0.01   # Minimum exploration rate
    epsilon_decay: float = 0.995 # Epsilon decay rate
    max_steps: int = 100        # Maximum steps per episode

class MazeEnvironment:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.setup_environment()
        self.setup_learning_parameters()
        self.setup_visualization()
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': []
        }
        
    def setup_environment(self):
        """Initialize the maze environment"""
        self.rows, self.cols = 6, 6
        self.maze = np.zeros((self.rows, self.cols))
        
        # Define maze elements (modified for better learning path)
        self.walls = [(1,1), (1,2), (3,0), (3,1), (2,3), (4,4)]
        self.traps = [(2,2), (4,1)]
        self.rewards = [(1,4), (4,2)]
        self.goal = (5, 5)
        self.start = (0, 0)
        
        # Set maze elements
        for wall in self.walls:
            self.maze[wall] = MazeElement.WALL.value
        for trap in self.traps:
            self.maze[trap] = MazeElement.TRAP.value
        for reward in self.rewards:
            self.maze[reward] = MazeElement.REWARD.value
        self.maze[self.goal] = MazeElement.GOAL.value
        self.maze[self.start] = MazeElement.START.value
        
        # Define actions
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def setup_learning_parameters(self):
        """Initialize Q-learning parameters"""
        self.Q = {}
        # Initialize Q-values for valid states only
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:  # Only initialize for non-wall states
                    self.Q[(r, c)] = {a: 0.0 for a in self.actions}
        
        self.epsilon = self.config.epsilon
        self.best_path = None
        self.best_reward = float('-inf')

    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(20, 6))
        self.fig.suptitle('Advanced Q-Learning Maze Navigation', fontsize=16)
        
        # Custom colormap for maze elements
        self.maze_colors = {
            MazeElement.EMPTY.value: 'darkgray',
            MazeElement.WALL.value: 'gray',
            MazeElement.GOAL.value: 'gold',
            MazeElement.START.value: 'green',
            MazeElement.TRAP.value: 'red',
            MazeElement.REWARD.value: 'cyan'
        }

    def get_reward(self, state: Tuple[int, int]) -> float:
        """Calculate reward for current state"""
        if state == self.goal:
            return 100.0
        elif state in self.walls:
            return -50.0
        elif state in self.traps:
            return -30.0
        elif state in self.rewards:
            return 20.0
        return -1.0  # Small negative reward to encourage finding shortest path

    def is_valid(self, state: Tuple[int, int]) -> bool:
        """Check if state is valid"""
        r, c = state
        return (0 <= r < self.rows and 
                0 <= c < self.cols and 
                state not in self.walls)

    def step(self, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float, bool]:
        """Execute one step in the environment"""
        move = self.action_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        
        if not self.is_valid(next_state):
            return state, self.get_reward(state), False
        
        done = next_state == self.goal
        return next_state, self.get_reward(next_state), done

    def visualize_maze(self, current_path: List[Tuple[int, int]] = None):
        """Visualize current maze state"""
        self.ax1.clear()
        self.ax1.set_title('Maze Environment\nEpisode: {}'.format(len(self.stats['episode_rewards'])))
        
        # Create maze visualization
        maze_plot = np.copy(self.maze)
        if current_path:
            for pos in current_path:
                if pos != self.goal and pos != self.start:
                    maze_plot[pos] = 0.5  # Path visualization
        
        # Custom colormap
        cmap = plt.cm.colors.ListedColormap(list(self.maze_colors.values()))
        bounds = list(self.maze_colors.keys())
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot maze
        self.ax1.imshow(maze_plot, cmap=cmap, norm=norm)
        
        # Add grid
        self.ax1.grid(True, which='major', color='black', linewidth=2)
        self.ax1.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
        self.ax1.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=element.name)
                         for element, color in zip(MazeElement, self.maze_colors.values())]
        self.ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    def visualize_q_values(self):
        """Visualize Q-values heatmap"""
        self.ax2.clear()
        self.ax2.set_title('Q-Values Heatmap')
        
        q_values = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    q_values[r, c] = max(self.Q[(r, c)].values())
        
        sns.heatmap(q_values, ax=self.ax2, cmap='viridis', annot=True, fmt='.1f')
        self.ax2.set_title('State Values\nMax Q-value: {:.2f}'.format(np.max(q_values)))

    def visualize_stats(self):
        """Visualize training statistics"""
        self.ax3.clear()
        self.ax3.set_title('Training Statistics')
        
        # Plot episode rewards
        rewards = np.array(self.stats['episode_rewards'])
        episodes = np.arange(len(rewards))
        self.ax3.plot(episodes, rewards, 'b-', label='Episode Reward', alpha=0.5)
        
        # Plot moving average
        window = 20
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            self.ax3.plot(episodes[window-1:], moving_avg, 'r-', 
                         label=f'Moving Average (n={window})', linewidth=2)
        
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Total Reward')
        self.ax3.legend()
        
        # Add success rate text
        if self.stats['success_rate']:
            success_rate = self.stats['success_rate'][-1]
            self.ax3.text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%',
                         transform=self.ax3.transAxes, verticalalignment='top')

    def train(self):
        """Train the agent using Q-learning"""
        try:
            for episode in range(self.config.episodes):
                state = self.start
                total_reward = 0
                current_path = [state]
                steps = 0
                
                while steps < self.config.max_steps:
                    # Epsilon-greedy action selection
                    if random.uniform(0, 1) < self.epsilon:
                        action = random.choice(self.actions)
                    else:
                        action = max(self.Q[state], key=self.Q[state].get)
                    
                    # Take action
                    next_state, reward, done = self.step(state, action)
                    total_reward += reward
                    
                    # Q-learning update
                    if next_state in self.Q:  # Only update if next_state is valid
                        best_next_value = max(self.Q[next_state].values())
                        self.Q[state][action] += self.config.alpha * (
                            reward + self.config.gamma * best_next_value - 
                            self.Q[state][action]
                        )
                    
                    state = next_state
                    current_path.append(state)
                    steps += 1
                    
                    if done:
                        break
                
                # Update statistics
                self.stats['episode_rewards'].append(total_reward)
                self.stats['episode_lengths'].append(steps)
                
                # Calculate success rate based on last 100 episodes
                success = done and total_reward > 0
                success_rate = (sum(1 for r in self.stats['episode_rewards'][-100:] 
                                  if r > 0) / min(100, episode + 1)) * 100
                self.stats['success_rate'].append(success_rate)
                
                # Update best path
                if success and total_reward > self.best_reward:
                    self.best_path = current_path
                    self.best_reward = total_reward
                
                # Visualize every 10 episodes
                if episode % 10 == 0:
                    self.visualize_maze(current_path)
                    self.visualize_q_values()
                    self.visualize_stats()
                    plt.pause(0.01)
                
                # Decay epsilon
                self.epsilon = max(self.config.min_epsilon, 
                                 self.epsilon * self.config.epsilon_decay)
            
            self.save_results()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_results()
    
    def save_results(self):
        """Save training results to file"""
        results = {
            'q_values': {str(k): v for k, v in self.Q.items()},
            'best_path': self.best_path,
            'best_reward': self.best_reward,
            'stats': self.stats
        }
        with open('maze_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to maze_training_results.json")

def main():
    # Configure training parameters
    config = TrainingConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        episodes=500,
        min_epsilon=0.01,
        epsilon_decay=0.995,
        max_steps=100
    )
    
    # Create and train the environment
    env = MazeEnvironment(config)
    env.train()
    
    # Show final plot
    plt.show()

if __name__ == "__main__":
    main()
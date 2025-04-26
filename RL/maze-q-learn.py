import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

class MazeEnvironment:
    def __init__(self):
        # Grid size
        self.rows, self.cols = 4, 4
        
        # Environment setup
        self.walls = [(1,1), (1,2), (3,0), (3,1)]  # Obstacle positions
        self.goal = (3, 3)                          # Goal position
        self.start = (0, 0)                         # Start position
        
        # Action space
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # Q-learning parameters
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.9      # Discount factor
        self.epsilon = 0.1    # Exploration rate
        self.episodes = 500   # Number of training episodes
        
        # Initialize Q-table
        self.Q = {(r, c): {a: 0 for a in self.actions} 
                 for r in range(self.rows) 
                 for c in range(self.cols)}
        
        # Visualization setup
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('Q-Learning Maze Navigation', fontsize=16)
        self.episode_rewards = []
        self.current_path = []
        
    def get_reward(self, state):
        """Define rewards for different states"""
        if state == self.goal:
            return 100        # High reward for reaching goal
        elif state in self.walls:
            return -100       # Penalty for hitting walls
        else:
            return -1         # Small penalty for each step to encourage efficiency
    
    def is_valid(self, state):
        """Check if a state is valid (within bounds and not a wall)"""
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols and state not in self.walls
    
    def step(self, state, action):
        """Execute one step in the environment"""
        move = self.action_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        if not self.is_valid(next_state):
            return state, self.get_reward(next_state)
        return next_state, self.get_reward(next_state)
    
    def visualize_maze(self):
        """Visualize the current state of the maze"""
        self.ax1.clear()
        self.ax1.set_title('Maze Environment')
        
        # Create maze grid
        maze_grid = np.ones((self.rows, self.cols))
        for wall in self.walls:
            maze_grid[wall] = 0
        
        # Plot maze with custom colors
        sns.heatmap(maze_grid, cmap='RdYlBu', ax=self.ax1, cbar=False)
        
        # Mark special positions
        self.ax1.plot(self.current_state[1] + 0.5, self.current_state[0] + 0.5, 'go', markersize=15, label='Agent')
        self.ax1.plot(self.goal[1] + 0.5, self.goal[0] + 0.5, 'r*', markersize=15, label='Goal')
        
        # Plot current path
        if self.current_path:
            path_y, path_x = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in self.current_path])
            self.ax1.plot(path_x, path_y, 'g--', linewidth=2, alpha=0.5)
        
        self.ax1.legend()
        
    def visualize_q_values(self):
        """Visualize Q-values for each state"""
        self.ax2.clear()
        self.ax2.set_title('Q-Values Heatmap')
        
        # Create Q-value grid
        q_values = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    q_values[r, c] = max(self.Q[(r, c)].values())
        
        sns.heatmap(q_values, ax=self.ax2, cmap='viridis', annot=True, fmt='.1f')
        
    def visualize_rewards(self):
        """Visualize training rewards"""
        self.ax3.clear()
        self.ax3.set_title('Training Rewards')
        self.ax3.plot(self.episode_rewards, 'b-')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Total Reward')
        
    def train(self):
        """Train the agent using Q-learning"""
        for episode in range(self.episodes):
            self.current_state = self.start
            total_reward = 0
            self.current_path = [self.current_state]
            
            while self.current_state != self.goal:
                # Epsilon-greedy action selection
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(self.actions)
                else:
                    action = max(self.Q[self.current_state], key=self.Q[self.current_state].get)
                
                # Take action and observe next state and reward
                next_state, reward = self.step(self.current_state, action)
                total_reward += reward
                
                # Q-learning update
                best_next_action = max(self.Q[next_state].values())
                self.Q[self.current_state][action] += self.alpha * (
                    reward + self.gamma * best_next_action - self.Q[self.current_state][action]
                )
                
                self.current_state = next_state
                self.current_path.append(self.current_state)
                
                # Visualize current state
                if episode % 10 == 0:  # Update visualization every 10 episodes
                    self.visualize_maze()
                    self.visualize_q_values()
                    self.visualize_rewards()
                    plt.pause(0.1)
            
            self.episode_rewards.append(total_reward)
            
            # Decay epsilon for better exploitation
            self.epsilon = max(0.01, self.epsilon * 0.995)

# Run the training
env = MazeEnvironment()
env.train()
print(row)

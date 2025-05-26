
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Button, HBox, Output
from IPython.display import display

class InteractiveGridWorld:
    def __init__(self):
        self.grid_size = 3
        self.goal = (2, 2)
        self.mountains = [(1, 1)]
        self.state_values = np.zeros((3, 3))
        self.policy = np.random.randint(0, 4, (3, 3))
        self.gamma = 0.9
        self.theta = 1e-3
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.output = Output()
        
        # Create buttons
        self.eval_btn = Button(description="Policy Evaluation")
        self.improve_btn = Button(description="Policy Improvement")
        self.reset_btn = Button(description="Reset")
        
        # Set up UI
        self.eval_btn.on_click(self.policy_evaluation)
        self.improve_btn.on_click(self.policy_improvement)
        self.reset_btn.on_click(self.reset_grid)
        self.controls = HBox([self.eval_btn, self.improve_btn, self.reset_btn])
        
        self.initialize_grid()
        self.update_visualization()

    def initialize_grid(self):
        self.grid = np.full((3, 3), -1.0)
        self.grid[self.goal] = 10
        for m in self.mountains:
            self.grid[m] = -2

    def move(self, state, action):
        i, j = state
        if action == 0:  # Up
            i = max(i-1, 0)
        elif action == 1:  # Down
            i = min(i+1, 2)
        elif action == 2:  # Left
            j = max(j-1, 0)
        elif action == 3:  # Right
            j = min(j+1, 2)
        return (i, j)

    def policy_evaluation(self, b=None):
        delta = 0
        for i in range(3):
            for j in range(3):
                v = self.state_values[i][j]
                action = self.policy[i][j]
                new_i, new_j = self.move((i,j), action)
                reward = self.grid[new_i][new_j]
                self.state_values[i][j] = reward + self.gamma * self.state_values[new_i][new_j]
                delta = max(delta, abs(v - self.state_values[i][j]))
        
        with self.output:
            print(f"Policy Evaluation Complete (Δ={delta:.4f})")
        self.update_visualization()

    def policy_improvement(self, b=None):
        policy_stable = True
        for i in range(3):
            for j in range(3):
                old_action = self.policy[i][j]
                q_values = []
                
                for action in range(4):
                    new_i, new_j = self.move((i,j), action)
                    reward = self.grid[new_i][new_j]
                    q = reward + self.gamma * self.state_values[new_i][new_j]
                    q_values.append(q)
                
                self.policy[i][j] = np.argmax(q_values)
                if old_action != self.policy[i][j]:
                    policy_stable = False
        
        with self.output:
            print("Policy Improvement Complete")
            if policy_stable:
                print("Policy has converged!")
        self.update_visualization()

    def reset_grid(self, b=None):
        self.state_values = np.zeros((3, 3))
        self.policy = np.random.randint(0, 4, (3, 3))
        with self.output:
            print("Grid World Reset")
        self.update_visualization()

    def update_visualization(self):
        self.ax.clear()
        self.ax.set_xticks(np.arange(3))
        self.ax.set_yticks(np.arange(3))
        self.ax.grid(which='both', color='black', linestyle='-', linewidth=2)
        
        # Draw values
        for i in range(3):
            for j in range(3):
                value = f"{self.state_values[i][j]:.1f}"
                self.ax.text(j, i, value, ha='center', va='center', fontsize=12)
                
                # Draw policy arrows
                action = self.policy[i][j]
                arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
                self.ax.text(j, i+0.3, arrow, ha='center', va='center', 
                           fontsize=16, color='red')
                
                # Color coding
                if (i,j) == self.goal:
                    color = 'lightgreen'
                elif (i,j) in self.mountains:
                    color = 'lightcoral'
                else:
                    color = 'lightblue'
                self.ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              facecolor=color, edgecolor='black'))
        
        self.ax.set_xlim(-0.5, 2.5)
        self.ax.set_ylim(-0.5, 2.5)
        self.fig.canvas.draw()

# To run the interactive grid world:
grid_world = InteractiveGridWorld()
display(grid_world.controls)
display(grid_world.output)

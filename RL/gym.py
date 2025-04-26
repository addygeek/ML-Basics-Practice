import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque
import time

class LunarLanderTrainer:
    def __init__(self):
        # Initialize environment
        self.env = gym.make("LunarLander-v2", render_mode="human")
        
        # Training parameters
        self.episodes = 1000
        self.max_steps = 1000
        self.window_size = 100  # For moving average
        
        # Statistics tracking
        self.episode_rewards = []
        self.moving_averages = []
        self.episode_lengths = []
        self.max_reward = float('-inf')
        self.min_reward = float('inf')
        
        # Setup visualization
        plt.style.use('dark_background')
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize the plotting environment"""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('LunarLander Training Statistics', fontsize=16)
        
        # Reward plot
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Total Reward')
        self.reward_line, = self.ax1.plot([], [], 'b-', label='Episode Reward')
        self.avg_reward_line, = self.ax1.plot([], [], 'r-', label='Moving Average')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Episode length plot
        self.ax2.set_title('Episode Lengths')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Steps')
        self.length_line, = self.ax2.plot([], [], 'g-')
        self.ax2.grid(True)
        
        # Success rate plot
        self.ax3.set_title('Success Rate')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Success Rate (%)')
        self.success_line, = self.ax3.plot([], [], 'y-')
        self.ax3.set_ylim(0, 100)
        self.ax3.grid(True)
        
        plt.tight_layout()
        plt.ion()  # Interactive mode on
        
    def update_plots(self, episode):
        """Update all plots with current statistics"""
        episodes = list(range(len(self.episode_rewards)))
        
        # Update reward plot
        self.reward_line.set_data(episodes, self.episode_rewards)
        if len(self.moving_averages) > 0:
            self.avg_reward_line.set_data(episodes[self.window_size-1:], self.moving_averages)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update episode length plot
        self.length_line.set_data(episodes, self.episode_lengths)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Update success rate plot
        if len(self.episode_rewards) >= self.window_size:
            success_rate = [sum(r > 200 for r in self.episode_rewards[max(0, i-self.window_size):i])/min(i, self.window_size)*100 
                          for i in range(1, len(self.episode_rewards)+1)]
            self.success_line.set_data(episodes, success_rate)
            self.ax3.relim()
            self.ax3.autoscale_view()
        
        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def calculate_moving_average(self):
        """Calculate moving average of rewards"""
        if len(self.episode_rewards) >= self.window_size:
            avg = np.convolve(self.episode_rewards, 
                            np.ones(self.window_size)/self.window_size, 
                            mode='valid')
            self.moving_averages = list(avg)
            
    def train(self):
        """Main training loop"""
        try:
            for episode in range(self.episodes):
                start_time = time.time()
                
                # Reset environment
                observation, _ = self.env.reset()
                total_reward = 0
                steps = 0
                
                # Episode loop
                for step in range(self.max_steps):
                    # For now, using random actions (replace with your RL algorithm)
                    action = self.env.action_space.sample()
                    
                    # Take action
                    observation, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                # Update statistics
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(steps)
                self.max_reward = max(self.max_reward, total_reward)
                self.min_reward = min(self.min_reward, total_reward)
                self.calculate_moving_average()
                
                # Update visualization
                if episode % 1 == 0:  # Update every episode
                    self.update_plots(episode)
                    
                # Print episode statistics
                print(f"Episode {episode + 1}/{self.episodes} | "
                      f"Steps: {steps} | "
                      f"Reward: {total_reward:.2f} | "
                      f"Time: {time.time() - start_time:.2f}s")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            self.env.close()
            plt.ioff()
            plt.show()
            
    def save_statistics(self):
        """Save training statistics to file"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'moving_averages': self.moving_averages,
            'max_reward': self.max_reward,
            'min_reward': self.min_reward
        }
        np.save('lunar_lander_stats.npy', stats)
        print("Statistics saved to lunar_lander_stats.npy")

def main():
    trainer = LunarLanderTrainer()
    trainer.train()
    trainer.save_statistics()

if __name__ == "__main__":
    main()

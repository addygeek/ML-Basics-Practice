# States: S0, S1, S2
# Actions: A0 (left), A1 (right)

import random

states = [0, 1, 2]
actions = [0, 1]  # 0 = left, 1 = right

# Rewards for reaching state 2
rewards = {2: 1}

# Define transition
def step(state, action):
    if action == 0:
        new_state = max(0, state - 1)
    else:
        new_state = min(2, state + 1)
    reward = rewards.get(new_state, 0)
    return new_state, reward

# Run simulation
state = 0
for i in range(10):
    action = random.choice(actions)
    new_state, reward = step(state, action)
    print(f"Step {i+1}: State {state} → Action {action} → State {new_state} → Reward: {reward}")
    state = new_state

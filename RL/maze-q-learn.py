import numpy as np
import random

# Grid size
rows, cols = 4, 4

# Walls (obstacles)
walls = [(1,1), (1,2), (3,0), (3,1)]
goal = (3, 3)
start = (0, 0)

# Rewards
def get_reward(state):
    if state == goal:
        return 10
    elif state in walls:
        return -100
    else:
        return -1

# Possible actions
actions = ['up', 'down', 'left', 'right']
action_map = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Check valid state
def is_valid(state):
    r, c = state
    return 0 <= r < rows and 0 <= c < cols and state not in walls

# Step function
def step(state, action):
    move = action_map[action]
    next_state = (state[0] + move[0], state[1] + move[1])
    if not is_valid(next_state):
        return state, get_reward(next_state)
    return next_state, get_reward(next_state)

# Q-table
Q = {}
for r in range(rows):
    for c in range(cols):
        Q[(r, c)] = {a: 0 for a in actions}

# Parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Exploration rate
episodes = 500

# Training loop
for ep in range(episodes):
    state = start
    while state != goal:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[state], key=Q[state].get)

        next_state, reward = step(state, action)
        best_next = max(Q[next_state], key=Q[next_state].get)

        # Q-learning update
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next] - Q[state][action])
        state = next_state

# Display learned policy
print("Learned Policy:")
for r in range(rows):
    row = ''
    for c in range(cols):
        if (r, c) in walls:
            row += ' XX '
        elif (r, c) == goal:
            row += ' GG '
        else:
            best_action = max(Q[(r, c)], key=Q[(r, c)].get)
            row += f' {best_action[0]}  '
    print(row)

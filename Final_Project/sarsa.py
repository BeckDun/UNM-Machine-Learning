from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import random
from utilities import CellType

'''
Train the agent using the SARSA reinforcement learning algorithm.

For each episode:
- Start the agent at a random free cell
- Choose an initial action using the current policy
- Repeatedly:
    - take an action
    - observe the next state and reward
    - choose the next action
    - update the Q-table

Training stops when:
- the target is reached
- or the maximum number of steps is exceeded

Unlike Q-learning, SARSA updates the Q-table using the
actual next action selected by the policy.
'''
def traning_loop(environment, agent, strategy, episodes=1000, max_steps=500):
    # Precompute free cells once so every episode can start from a random position.
    # This gives the Q-table coverage across the full state space, not just
    # cells reachable from one fixed start.
    free_cells = [
        (r, c)
        for r in range(environment.map.shape[0])
        for c in range(environment.map.shape[1])
        if environment.map[r, c] == CellType.EMPTY.value
    ]

    for _ in tqdm(range(episodes)):
        steps = 0
        state = random.choice(free_cells)
        action = agent.choose_action(state)
        done = False

        while not done and steps < max_steps:
            next_state, next_action, _, done = agent.step(state, action, strategy, method="sarsa")
            steps += 1
            state = next_state

            if not done:
                action = next_action
                
    
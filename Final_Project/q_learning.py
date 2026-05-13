from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import random
from utilities import CellType

'''
Train the agent by repeatedly interacting with the environment
and updating the Q-table.

For each episode:
- Start the agent in a random free cell
- Let the agent move through the environment
- Update the Q-table after every action
- Stop when:
    - the target is reached
    - or max_steps is exceeded
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
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, _, _, done = agent.step(state, action, strategy, method="q_learning")
            steps += 1
            state = next_state
            

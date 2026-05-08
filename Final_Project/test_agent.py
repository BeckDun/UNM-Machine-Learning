import numpy as np
from matplotlib import pyplot as plt

def test_agent(environment, agent, strategy, max_steps=200):
    agent.epsilon = 0
    state = environment.agent_start
    path = [state]
    done = False
    steps = 0

    plt.figure(figsize=(8, 8))

    while(not done and steps < max_steps):
        environment.plot_environment(environment.map, state)
        row, col = state

        best_action_index = np.argmax(agent.q_table[row, col])
        action = agent.actions_list[best_action_index]

        next_state, _, done = environment.move(state, action, strategy)

        path.append(next_state)
        state = next_state
        steps += 1
    
    environment.plot_environment(environment.map, state)
    plt.show()


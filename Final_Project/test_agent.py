import numpy as np
from matplotlib import pyplot as plt
from utilities import CellType

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
    plt.savefig("plots/fig.png")
    plt.show()


def test_accuracy(environment, agent, strategy, max_steps=200):
   
    old_epsilon = agent.epsilon
    agent.epsilon = 0

    valid_starts = [
        (r, c)
        for r in range(environment.map.shape[0])
        for c in range(environment.map.shape[1])
        if environment.map[r, c] == CellType.EMPTY.value
    ]

    valid_count = 0

    for start in valid_starts:
        state = start
        done = False
        steps = 0

        while not done and steps < max_steps:
            row, col = state
            best_action_index = np.argmax(agent.q_table[row, col])
            action = agent.actions_list[best_action_index]
            state, _, done = environment.move(state, action, strategy)
            steps += 1

        if done:
            valid_count += 1

    agent.epsilon = old_epsilon

    total = len(valid_starts)
    accuracy = (valid_count / total * 100) if total > 0 else 0.0
    print("    accuracy: ", accuracy, ", valid_count: ", valid_count, ", total: ", total)
    return accuracy, valid_count, total


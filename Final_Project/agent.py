import numpy as np
from environment import Move
from environment import Environment
class Agent:

    def __init__ (self, enviroment, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
        self.enviroment = enviroment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        height, width = enviroment.map.shape
        self.actions_list = list(Move)

        # Q-table is indexed as [row][col][action_index]
        self.q_table = np.zeros((height, width, len(self.actions_list)))

    '''
    This function picks an action for the agent to take using an epsilon-greedy policy.
    An epsilong-gready strategy is when we use epsilon as the probability of choosing a
    random action instead of the current optimal action. 

    Args:
        state (tuple[int,int]): The (row, column) position in the enviroment

    Return:
        Move (Enum): The selected action

    '''
    def choose_action(self, state: tuple[int,int]):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions_list)
        else:
            row, col = state
            q_values = self.q_table[row, col]

            max_q = np.max(q_values)
            best_indices = np.where(q_values == max_q)[0]
            chosen_index = np.random.choice(best_indices)
            return self.actions_list[chosen_index]


    def step(self, state: tuple[int,int], action: Move, strategy, method = "q_learning"):
        action_index = self.actions_list.index(action)

        next_state, reward, done = self.enviroment.move(state, action, strategy)

        row, col = state
        next_row, next_col = next_state

        if done:
            target = reward
            next_action = None
        else:
            if(method == "q_learning"):
                best_next = np.max(self.q_table[next_row, next_col])
                target = reward + self.gamma * best_next
                next_action = None
            elif(method == "sarsa"):
                next_action = self.choose_action(next_state)
                next_action_index = self.actions_list.index(next_action)
                target = reward + self.gamma * self.q_table[next_row, next_col, next_action_index]
            
        self.q_table[row, col, action_index] += self.alpha * (target - self.q_table[row, col, action_index])

        return next_state, next_action, reward, done




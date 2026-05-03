# Strategy class

import numpy as np
from abc import ABC, abstractmethod
from utilities import CellType, block_at

class Strategy(ABC):

    @abstractmethod
    def get_reward():
        pass
        

class NaiveStrategy(Strategy):

    def __init__(self, target_reward : int = 100, 
                 punishment : int = -100, 
                 neutral_action : int = 0):
        self.target_reward = target_reward
        self.punishment = punishment
        self.neutral_action = neutral_action

    def get_reward(self, map : np.NDArray, curr_state : tuple[int, int]) -> int:

        match block_at(map, curr_state):
            case CellType.EMPTY:
                return self.neutral_action
            case CellType.TARGET:
                return self.target_reward
            case _:
                return self.punishment
                


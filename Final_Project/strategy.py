# Strategy class

import numpy as np
from utilities import CellType, block_at


class NaiveStrategy():
    def __init__(self, target_reward : int = 100,
                 oob_punishment : int = -100,
                 obstacle_punishment: int = -100,
                 neutral_action : int = -1):
        self.target_reward = target_reward
        self.oob_punishment = oob_punishment
        self.neutral_action = neutral_action
        self.obstacle_punishment = obstacle_punishment

    def get_reward(self, map : np.ndarray, curr_state : tuple[int, int]) -> int:

        match block_at(map, curr_state):
            case CellType.EMPTY.value:
                return self.neutral_action
            case CellType.TARGET.value:
                return self.target_reward
            case CellType.OBSTACLE.value:
                return self.obstacle_punishment
            case _:
                return self.oob_punishment


class DistanceShapedStrategy():
    def __init__(self,
                 target: tuple[int, int],
                 target_reward: int = 100,
                 oob_punishment: int = -100,
                 obstacle_punishment: int = -100,
                 distance_scale: float = 1.0):
        self.target = target
        self.target_reward = target_reward
        self.oob_punishment = oob_punishment
        self.obstacle_punishment = obstacle_punishment
        self.distance_scale = distance_scale

    def get_reward(self, map: np.ndarray, curr_state: tuple[int, int]) -> float:
        cell = block_at(map, curr_state)

        if cell == CellType.TARGET.value:
            return self.target_reward
        elif cell == CellType.OBSTACLE.value:
            return self.obstacle_punishment
        elif cell == CellType.OOB.value:
            return self.oob_punishment
        else:
            dist = (abs(curr_state[0] - self.target[0]) +
                              abs(curr_state[1] - self.target[1]))
            return -self.distance_scale * dist

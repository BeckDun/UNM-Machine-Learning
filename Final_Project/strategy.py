# Strategy class

import numpy as np
from utilities import CellType, block_at


class NaiveStrategy():
    """
    A simple strategy that provides rewards solely based on the state the agent just arrived in. 
    A reward is provided for reaching the target and a strong punishment is provided for colliding 
    with an obstacle or moving out of bounds. 
    To incentivize reaching the target, a small penalty is provided when the agent's next state is an 
    empty cell. 
    """
    def __init__(self, target_reward : int = 100,
                 oob_punishment : int = -100,
                 obstacle_punishment: int = -100,
                 neutral_action : int = -1):
        self.target_reward = target_reward
        self.oob_punishment = oob_punishment
        self.neutral_action = neutral_action
        self.obstacle_punishment = obstacle_punishment

    def get_reward(self, map : np.ndarray, curr_state : tuple[int, int]) -> int:
        """
        Returns the integer reward based on the agent's new state and the strategy defined in the initializer.
        """

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
    """
    Instead of giving the same small penalty for every empty cell (like S1 does),
    this strategy penalizes the agent more the farther it is from the target.
    The penalty is based on Manhattan distance — the number of horizontal plus
    vertical steps between the agent and the target. So a cell right next to the
    target gets a reward close to -1, while a cell far away might get -30 or worse.

    This creates a natural gradient: moving toward the target always feels better
    than moving away, even if the agent hasn't found the target yet.

    Terminal rewards (reaching target, hitting obstacle, going out of bounds)
    are the same as the Naive strategy.
    """

    def __init__(self,
                 target: tuple[int, int],
                 target_reward: int = 100,
                 oob_punishment: int = -100,
                 obstacle_punishment: int = -100,
                 distance_scale: float = 1.0):
        # Where the target is on the map — needed to compute distance each step
        self.target = target
        self.target_reward = target_reward
        self.oob_punishment = oob_punishment
        self.obstacle_punishment = obstacle_punishment
        
        self.distance_scale = distance_scale

    def get_reward(self, map: np.ndarray, curr_state: tuple[int, int]) -> float:
        cell = block_at(map, curr_state)

        # reward for reaching the goal
        if cell == CellType.TARGET.value:
            return self.target_reward

        #  penalty for walking into a wall
        elif cell == CellType.OBSTACLE.value:
            return self.obstacle_punishment

        #  penalty for stepping outside the map boundary
        elif cell == CellType.OOB.value:
            return self.oob_punishment

        # For any normal empty cell, penalize based on how far the agent still is.
        # Manhattan distance = |row difference| + |column difference|
        else:
            dist = (abs(curr_state[0] - self.target[0]) +
                    abs(curr_state[1] - self.target[1]))
            return -self.distance_scale * dist

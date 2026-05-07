# RL Environment

from map_abstraction import MapAbstraction
import numpy as np
from matplotlib import pyplot as plt

from enum import Enum
from utilities import CellType, block_at
import random

class Move(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

class Environment:

    """
    RLEnvironment handles the following: 
    - Simulating the moves in the abstracted environment
    - init takes in: 
      - map abstraction
      - target position
    - function for agent interactions
     - takes in: current state and action
     - returns: next state and immediate_reward

     reward will be tied with the *resulting state", not the movement action that 
     brought the agent to that state. 

     0 is a free space, 1 is an obstacle. 
    """

    def __init__(self,
                 bmp_file = None, 
                 map = None,
                 target = None,
                 agent_start : tuple[int, int] = (0, 0)):
        
        if (bmp_file is None and map is None):
            raise ValueError("Must pass in either a bitmap or a created map!")

        if map is None:
            abstractor = MapAbstraction(bmp_file)
            self.map = abstractor.get_abstract_map((40, 40))
        else:
            self.map = map

        if target:
            self.map[target[0]][target[1]] = CellType.TARGET.value
            self.target = target
        else:
            self.target = self.choose_target(self.map)

        self.agent_start = agent_start
            
    def is_valid_state(self, state: tuple[int, int]) -> bool:
        row, col = state
        height, width = self.map.shape

        if row < 0 or row >= height:
            return False

        if col < 0 or col >= width:
            return False

        if block_at(self.map, state) == CellType.OBSTACLE.value:
            return False

        return True

    def choose_target(self, map):
        """
        Choose a random empty block on the map as the target.

        Params 
        -----------------------------------------
        map: a 2D numpy array of the map. 

        Return 
        ------------------------------------------
        None

        """
        free_cells = [(row, col) 
                      for row in range(np.size(map, 0))
                      for col in range(np.size(map, 1))
                      if block_at(map, (row, col)) == CellType.EMPTY.value]
        
        loc =  random.choice(free_cells)
        map[loc[0]][loc[1]] = CellType.TARGET.value
        return loc


    def move(self, old_state : tuple, action : Move, strategy) -> tuple:
        """
        return:
        - new state (int tuple of location)
        - reward (int)
        """

        candidate_state = (old_state[0] + action.value[0], old_state[1] + action.value[1])

        reward = strategy.get_reward(map = self.map, curr_state=candidate_state)

        if self.is_valid_state(candidate_state):
            new_state = candidate_state
        else:
            new_state = old_state

        
        done = block_at(self.map, curr_state=new_state) == CellType.TARGET.value

        return new_state, reward, done
    
    def plot_environment(self, map : np.ndarray, current_state : tuple) -> None:
        plt.clf()

        # Display the 2D array. 'cmap="gray_r"' makes 0 white (free) and 1 black (obstacle).
        # Change it to 'cmap="gray"' if you want 0 to be black and 1 to be white.
        plt.imshow(map, cmap='gray_r', origin='upper')

        # Add gridlines to show the discrete grid blocks clearly
        plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.xticks(np.arange(-.5, 40, 1), []) # Hide labels, keep grid ticks matching shape
        plt.yticks(np.arange(-.5, 40, 1), []) 

        plt.plot(current_state[1], current_state[0], "or")
        plt.plot(self.target[1], self.target[0], "oy")

        plt.title("Current Map")
        plt.pause(0.05)

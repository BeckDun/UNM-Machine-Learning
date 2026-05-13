# RL Environment

from map_abstraction import MapAbstraction
import numpy as np
from matplotlib import pyplot as plt

from enum import Enum
from utilities import CellType, block_at
import random

"""
A utility enum to define how the agent moves. 
"""
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
        """
        Initialize the environment. Assign arguments to this instance if necessary. 
        Construct the map if necessary and set the agent's starting location. 
        """
        
        # either pass in a bitmap to make an abstracted file, or pass in an alread abstracted map. 
        if (bmp_file is None and map is None):
            raise ValueError("Must pass in either a bitmap or a created map!")

        if map is None:
            abstractor = MapAbstraction(bmp_file)
            self.map = abstractor.get_abstract_map((40, 40))
        else:
            self.map = map

        # if the target is not specified, select a random target on an empty cell in the map. 
        if target:
            self.map[target[0]][target[1]] = CellType.TARGET.value
            self.target = target
        else:
            self.target = self.choose_target(self.map)

        # assign to the instance the starting coordinates of the agent. 
        self.agent_start = agent_start
            
        
    def is_valid_state(self, state: tuple[int, int]) -> bool:
        """
        Verifies that the state (location, in this case) allows for the agent to continue forward. 
        Valid states include reaching the target or moving to an empty cell.
        Invalid states include navigating out of bounds or colliding with an obstacle. 

        Return
        ________
        A boolean. True if within bounds, False, otherwise. 
        
        """   
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
        ______
        map: a 2D numpy array representing the abstracted map. 
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
        Given the current state of the agent and an action to perform (selected at random by the agent), 
        move the agent to the next state. 

        Params
        ______
        self: this class instance. 

        old_state: a 2-tuple of the x-y coordinates composing the agent's location. 

        action: A Move (from the Move enum class) with the x and y offsets that, when added
        to the current state, give the next state. 

        strategy: The Strategy to employ (either Naive or Manhattan) that returns the immediate reward
        obtained by moving to the new state. 

        Returns
        ______
        A 3-tuple consisting of:
        - the new state, a 2-tuple of the x and y coordinate representing the agent's location. 
        - the reward, an integer value that either incentivizes or punishes the agent. 
        - A boolean flag 'done', indicating to the agent whether they have reached the obstacle. 

        """

        # calculate the future state that the agent wants to move in. 
        candidate_state = (old_state[0] + action.value[0], old_state[1] + action.value[1])

        # calculate the reward based on that movement. 
        reward = strategy.get_reward(map = self.map, curr_state=candidate_state)

        # check if the new state is valid. 
        if self.is_valid_state(candidate_state):
            new_state = candidate_state
        else:
            new_state = old_state

        # calculate whether the block at the given location is the target. 
        done = block_at(self.map, curr_state=new_state) == CellType.TARGET.value

        return new_state, reward, done
    
    def plot_environment(self, map : np.ndarray, current_state : tuple) -> None:
        """
        A utility class to visualize the agent's location on the map

        Params
        ______
        self: an instance of this class.

        map: a 2D numpy array of the abstracted map. 

        current state: a 2-tuple of the x and y coordinate of the agent's location. 

        """
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

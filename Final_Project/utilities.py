# shared utilities. 

from enum import Enum
import numpy as np

class CellType(Enum):
    """
    A utility enum class to assign integer values to various states. 
    """
    OOB = -1
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2

def block_at(map : np.ndarray, curr_state : tuple[int, int]) -> int:
        """
        A utility method that returns the integer representation of the cell the agent is on
        as defined by the CellType enum. 

        Params
        ________
        map: a 2D numpy array of the abstracted map. 
        curr_state: the *new state* that the agent has just arrived at, as opposed to the 
        old or previous state. A 2-tuple of the x and y coordinates that compose the agent's location. 

        Returns
        ________
        An integer corresponding to the the cell type at the provided location. 

        """
        height = np.size(map, 0)
        width = np.size(map, 1)

        if(
            curr_state[0] < 0 or curr_state[0] >= height or
            curr_state[1] < 0 or curr_state[1] >= width
            ):
            return CellType.OOB.value
        else:
            return map[curr_state[0]][curr_state[1]] 
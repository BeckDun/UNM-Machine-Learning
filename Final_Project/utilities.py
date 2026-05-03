# shared utilities. 

from enum import Enum
import numpy as np

class CellType(Enum):
    OOB = -1,
    EMPTY = 0,
    OBSTACLE = 1,
    TARGET = 2

def block_at(map : np.NDArray, curr_state : tuple[int, int]) -> int:
        height = np.size(map, 0)
        width = np.size(map, 1)

        if(
            curr_state[0] < 0 or curr_state[0] >= height or
            curr_state[1] < 0 or curr_state[1] >= width
            ):
            return CellType.OOB
        else:
            return map[curr_state[0]][curr_state[1]] 
# map abstraction
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt


class MapAbstraction:
    '''
    This class abstracts bmp files of maps to be a regularized and smaller sized maps, stored in numpy arrays. 
    We use a simple pooling strategy to get the average color in a given area and reduce the size, assigning the color black or white. 
    This allows the agent to have a smaller map to explore and exports the map for easy usage in the rest of the programs. 
    ''' 
    def __init__(self, bmp_file):
        self.bmp_file = bmp_file
    
    def load_binary_map(self, limit=128):
        img = Image.open(self.bmp_file).convert('L')
        arr = np.array(img)
        self.binary_map = (arr < limit).astype(np.uint8)
        return self.binary_map
    
    def abstract_map(self, binary_map, new_shape):
        height, width = binary_map.shape
        new_h, new_w = new_shape

        out = np.zeros((new_h, new_w), dtype=np.uint8)

        for i in range(new_h):
            # Map output row i back to a range of source rows.
            # max(..., r0+1) guarantees the block is never empty when sampling
            # (e.g. abstracting a 20x20 map to 40x40).
            r0 = int(i * height / new_h)
            r1 = max(r0 + 1, int((i + 1) * height / new_h))
            r1 = min(r1, height)

            for j in range(new_w):
                c0 = int(j * width / new_w)
                c1 = max(c0 + 1, int((j + 1) * width / new_w))
                c1 = min(c1, width)

                # Over-approximation: if any source pixel is an obstacle, mark as obstacle
                out[i, j] = 1 if binary_map[r0:r1, c0:c1].any() else 0

        return out


    def get_abstract_map(self, new_shape, threshold=128):
        binary_map = self.load_binary_map(threshold)
        return self.abstract_map(binary_map, new_shape)
    

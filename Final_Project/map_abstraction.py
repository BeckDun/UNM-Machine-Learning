# map abstraction
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt


class MapAbstraction: 
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

        
        # pooling to get new map
        bh = math.ceil(height/new_h)
        bw = math.ceil(width/new_w)
        out = np.zeros((new_h, new_w), dtype=np.uint8)

        for i in range(new_h):
            r0 = i * bh
            r1 = min((i+1)*bh, height)
            for j in range(new_w):
                c0 = j * bw
                c1 = min((j+1)* bw, width)
                
                block = binary_map[r0:r1 , c0:c1]
                
                out[i,j] = 0 if block.size == 0 else (1 if block.max() else 0)
        return out


    def get_abstract_map(self, new_shape, threshold=128):
        binary_map = self.load_binary_map(threshold)
        return self.abstract_map(binary_map, new_shape)
    

# x = MapAbstraction('map4.bmp')


# abstracted = x.get_abstract_map((40, 40))

# # --- MATPLOTLIB DISPLAY ---

# # Set up the figure size
# plt.figure(figsize=(8, 8))

# # Display the 2D array. 'cmap="gray_r"' makes 0 white (free) and 1 black (obstacle).
# # Change it to 'cmap="gray"' if you want 0 to be black and 1 to be white.
# plt.imshow(abstracted, cmap='gray_r', origin='upper')

# # Add gridlines to show the discrete grid blocks clearly
# plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
# plt.xticks(np.arange(-.5, 40, 1), []) # Hide labels, keep grid ticks matching shape
# plt.yticks(np.arange(-.5, 40, 1), []) 

# plt.title("Abstracted Map (40x40 Grid)")
# plt.show()
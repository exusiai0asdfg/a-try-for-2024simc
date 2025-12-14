import numpy as np
from rotation import rotation

def get_rotations(img_flat, dim):
    img_2d = img_flat.reshape(dim, dim)
    rots = []
    for k in range(4):
        rots.append(rotation(img_2d, k).flatten())
    return np.array(rots)
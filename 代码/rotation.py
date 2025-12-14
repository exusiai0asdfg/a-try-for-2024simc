import numpy as np
def rotation(img, k):
    return np.rot90(img, k=-k)
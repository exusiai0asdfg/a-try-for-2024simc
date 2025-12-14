import numpy as np
import matplotlib.pyplot as plt
from load import load
from rotation import rotation
from show import save_plot

def task2(data):
    patterns_flat = data
    dim = 36
    
    r_flat = patterns_flat[0]
    r_img = r_flat.reshape(dim, dim)
    save_plot(r_img, "Task 2 ", "task2.png", cmap='gray')
    
    rs_flat = {}
    for k, angle in enumerate([0, 90, 180, 270]):
        rot_img = rotation(r_img, k)
        rs_flat[angle] = rot_img.flatten()
        
    counts = {0: 0, 90: 0, 180: 0, 270: 0}
    for p in patterns_flat:
        for angle, r_vec in rs_flat.items():
            if np.allclose(p, r_vec):
                counts[angle] += 1
                break
    print(f"Task 2 Counts: {counts}")


if __name__ == "__main__":
    data2=load(2)
    task2(data2)
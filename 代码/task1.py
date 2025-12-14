import numpy as np
import matplotlib.pyplot as plt
from load import load
from rotation import rotation


def task1(data):
    patterns = data
    r_img = patterns[0]
    target_angles = [0, 90, 180, 270]
    rs = {}
    for k, angle in enumerate(target_angles):
        rotated_image = rotation(r_img, k)
        rs[angle] = rotated_image  

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, (angle, img) in zip(axes, rs.items()):
        ax.imshow(img, cmap='Blues')
        ax.set_title(f"{angle} deg")
        ax.axis('off')
    plt.tight_layout() 
    plt.savefig('task1_rotations.png')
    plt.close()
    
    counts = {0: 0, 90: 0, 180: 0, 270: 0}
    for p in patterns:
        for angle, r_img in rs.items():
            if np.allclose(p, r_img, atol=1e-5): 
                counts[angle] += 1
                break
    print(f"Task 1 Counts: {counts}")
    return counts

if __name__ == "__main__":
    data1=load(1)
    task1(data1)
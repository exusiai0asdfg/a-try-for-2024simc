import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from load import load
from rotation import rotation
from show import save_plot

def task6(rows, cols):
    n_patterns = 65535
    dim = 25
    n_pixels = dim * dim
    
    vals = np.ones(len(rows))
    
    x_sparse = csr_matrix((vals, (rows, cols)), shape=(n_patterns, n_pixels))
    patterns = x_sparse.toarray()
    
    avg_row_sum = np.mean(np.sum(patterns, axis=1))
    print(f"Task 6a Avg Row Sum: {avg_row_sum}")
    
    avg_flat = np.mean(patterns, axis=0)
    save_plot(avg_flat.reshape(dim, dim), "Task 6c", "task6c.png", cmap='viridis')
    

    kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=1024, n_init=3)
    labels = kmeans.fit_predict(patterns)
    
    centers = []
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    for i in range(4):
        center_img = kmeans.cluster_centers_[i].reshape(dim, dim)
        centers.append(center_img)
        axes[i].imshow(center_img, cmap='viridis')
        axes[i].set_title(f"Class {i}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('task6c.png')
    plt.close()
    print("Saved task6c.png")

    final_img = centers[0].copy()
    for i in range(1, 4):
        candidate = centers[i]
        dists = []
        rots = []
        for k in range(4):
            r_img = rotation(candidate, k)
            rots.append(r_img)
            dists.append(np.linalg.norm(final_img - r_img))
        best_k = np.argmin(dists)
        final_img += rots[best_k]

    save_plot(final_img / 4, "Task 6 Re", "task6_re.png", cmap='viridis')

if __name__ == "__main__":
    rows6 = load("6a")
    cols6 = load("6b")
    task6(rows6, cols6)

"""
思路：
首先考虑没光0比较多，先压成csr再换成numpy，6c用minibatch来算kmeans，再重复task4思路即可
"""
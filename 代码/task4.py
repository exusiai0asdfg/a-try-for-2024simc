import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from load import load
from rotation import rotation
from show import save_plot

def task4(data):
    patterns = data
    dim = 50
    
    #4a
    row_sums = np.sum(patterns, axis=1)
    avg_row_sum = np.mean(row_sums)
    print(f"Task 4a :{avg_row_sum}")
    
    #4b
    avg_flat = np.mean(patterns, axis=0)
    avg_img = avg_flat.reshape(dim, dim)
    save_plot(avg_img, "Task 4b ", "task4b.png", cmap='viridis')
    
    #4c
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(patterns)
    
    centers = []
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    for i in range(4):
        center_flat = kmeans.cluster_centers_[i]
        center_img = center_flat.reshape(dim, dim)
        centers.append(center_img)
        
        axes[i].imshow(center_img, cmap='viridis')
        axes[i].set_title(f"Class {i}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('task4c.png')
    plt.close()
    print("Saved task4c.png")

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
    
    final_img /= 4
    save_plot(final_img, "Task 4 re", "task4_re.png", cmap='viridis')

if __name__ == "__main__":
    data4 = load(4)
    task4(data4)



"""
思路：
4c：
依然使用K-Means，分出四个聚类，再取聚类中心，把四个聚类中心一转，叠加即可
“”“
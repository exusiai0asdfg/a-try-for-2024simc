import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from show import save_plot
from load import load



def task3(data):
    patterns = data
    dim = 33
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(patterns)
    
    idx = np.argsort(labels)
    sorted_patterns = patterns[idx]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(patterns, cmap='BuPu', aspect='auto')
    plt.title("Unsorted")
    plt.subplot(1, 2, 2)
    plt.imshow(sorted_patterns, cmap='BuPu', aspect='auto')
    plt.title("Sorted by Cluster")
    plt.savefig('task3_matrix.png')
    plt.close()
    
    center_img = kmeans.cluster_centers_[0].reshape(dim, dim)
    save_plot(center_img, "Task 3", "task3.png", cmap='BuPu')

if __name__ == "__main__":
    data3=load(3)
    task3(data3)

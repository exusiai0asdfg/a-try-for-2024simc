import numpy as np
from scipy.sparse import csr_matrix
from load import load
from rotation import rotation
from show import save_plot
from get_rotations import get_rotations


def task7(rows, cols):

    n_patterns = 100000
    dim = 25
    n_pixels = dim * dim
    
    vals = np.ones(len(rows))
    X_sparse = csr_matrix((vals, (rows, cols)), shape=(n_patterns, n_pixels))
    patterns = X_sparse.toarray()

    model = np.mean(patterns, axis=0)
    
    iterations = 10
    for i in range(iterations):
        
        model_rots = get_rotations(model, dim)
        
        correlations = patterns @ model_rots.T
        best_rots_idx = np.argmax(correlations, axis=1)    
        new_model = np.zeros_like(model)
        
        for r_k in range(4):
            mask = (best_rots_idx == r_k)
            count = np.sum(mask)
            
            if count > 0:
                subset = patterns[mask]
                
                subset_sum_flat = np.sum(subset, axis=0)
                subset_sum_img = subset_sum_flat.reshape(dim, dim)
                inverse_k = (4 - r_k) % 4
                aligned_img = rotation(subset_sum_img, inverse_k)
                
                new_model += aligned_img.flatten()
        
        model = new_model / n_patterns
    

    final_img = model.reshape(dim, dim)
    save_plot(final_img, "Task 7 final reconstruction", "task7_final.png", cmap='viridis')
    print("reconstruction complete. Saved task7_final.png")

if __name__ == "__main__":
    rows7 = load("7a")
    cols7 = load("7b")
    task7(rows7, cols7)



"""
思路：
采取迭代、对齐、更新的想法，先取平均来获得一个rot，接下来进行10次迭代（iterations），将同方向取出重组，对齐累加获得信rot，循环之后应该收敛得到头像
"""
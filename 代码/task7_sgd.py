import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from load import load
from show import save_plot
from rotation import rotation
from get_rotations import get_rotations


def solve_sgd(patterns, dim, n_epochs=20, batch_size=2048, lr=0.1):

    n_samples,_ = patterns.shape
    

    master_image = patterns.mean(axis=0)
    
    loss_history = []
    print(f"Start SGD Training: Samples={n_samples}, Epochs={n_epochs}, Batch={batch_size}, LR={lr}")

    for epoch in range(n_epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_X = patterns[batch_idx]
            current_bs = batch_X.shape[0]
            

            master_rots = get_rotations(master_image, dim)
            
            diffs = batch_X[:, np.newaxis, :] - master_rots[np.newaxis, :, :]
            dists_sq = np.sum(diffs**2, axis=2) 
            
            best_rot_indices = np.argmin(dists_sq, axis=1) 
            

            min_dists = dists_sq[np.arange(current_bs), best_rot_indices]
            epoch_loss += np.mean(min_dists)
            n_batches += 1
            
            grad_accumulator = np.zeros_like(master_image)
            
            for r in range(4):
                mask = (best_rot_indices == r)
                if np.sum(mask) == 0: continue
                

                X_subset = batch_X[mask]
                
                residuals = X_subset - master_rots[r]
                mean_grad = residuals.mean(axis=0)
                
                grad_2d = mean_grad.reshape(dim, dim)
                
                grad_rotated_back = rotation(grad_2d, -r).flatten()
                
                weight = np.sum(mask) / current_bs
                grad_accumulator += grad_rotated_back * weight
            

            master_image += lr * grad_accumulator


        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

    return master_image, loss_history

def task7_sgd(rows, cols):
    print("Processing Task 7 using SGD (Neural Approach)...")
    

    n_patterns = 100000
    dim = 25
    n_pixels = dim * dim
    
    vals = np.ones(len(rows))
    X_sparse = csr_matrix((vals, (rows, cols)), shape=(n_patterns, n_pixels))
    patterns = X_sparse.toarray()
    

    final_img_flat, history = solve_sgd(patterns, dim, n_epochs=15, batch_size=2048, lr=0.1)
    

    plt.figure()
    plt.plot(history, marker='o')
    plt.title("SGD Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("task7_sgd_loss.png")
    plt.close()
    print("Saved task7_sgd_loss.png")
    
    final_img = final_img_flat.reshape(dim, dim)
    save_plot(final_img, "Task 7 SGD Reconstruction", "task7_sgd_result.png", cmap='viridis')

if __name__ == "__main__":
    rows7 = load("7a")
    cols7 = load("7b")
    task7_sgd(rows7, cols7)
"""
曾经看过一点深度学习和神经网络的书，不是很熟，也没学过pytorch，在ai帮助下搓了个简单的随机梯度下降的神经网络出来，感觉还可以，就留着了。
"""
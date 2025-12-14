import matplotlib.pyplot as plt
def save_plot(data, title, fname, cmap='viridis'):
    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
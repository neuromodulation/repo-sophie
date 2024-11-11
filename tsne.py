from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import mne


def load_brainvision_file(filepath):
        raw = mne.io.read_raw_brainvision(filepath, preload=True)
        available_channels = raw.ch_names
        ecogs = {}
        for channel in available_channels:
            if "ECOG" not in channel and "LFP" not in channel:
                continue
            data, _ = raw[channel, :]
            data = torch.tensor(data, dtype=torch.float32).squeeze()
            ecogs[channel] = data 
        return data

X = load_brainvision_file("data\sub-000_ses-right_task-force_run-0_ieeg.vhdr").reshape(-1, 1)

def apply_tsne_and_plot_raw(X, n_components=1, perplexity=30, random_state=0): #n dimensions, 
    """
    Applies t-SNE to raw data and creates a scatter plot.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The data to be reduced.
    n_components : int, default=2
        Number of dimensions for t-SNE (2 or 3 for visualization).
    perplexity : float, default=30
        Balances local vs. global structure. Larger datasets may need higher values.
    random_state : int, default=0
        Seed for reproducibility.

    Returns:
    None
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # Plot in 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=10)
    plt.title("t-SNE Visualization of Raw Data")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

apply_tsne_and_plot_raw(X)  # Where X is your raw high-dimensional dataset


# import umap

# def apply_umap_and_plot(X, labels=None, n_components=2):
#     reducer = umap.UMAP(n_components=n_components, random_state=42)
#     X_umap = reducer.fit_transform(X)
    
#     plt.figure(figsize=(8, 6))
#     if labels is not None:
#         plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='viridis', alpha=0.7)
#         plt.colorbar(label='Cluster Label')
#     else:
#         plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7)
#     plt.title("UMAP Visualization")
#     plt.xlabel("UMAP 1")
#     plt.ylabel("UMAP 2")
#     plt.show()

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import torch
# import mne


# def load_brainvision_file(filepath):
#         raw = mne.io.read_raw_brainvision(filepath, preload=True)
#         available_channels = raw.ch_names
#         ecogs = {}
#         for channel in available_channels:
#             if "ECOG" not in channel and "LFP" not in channel:
#                 continue
#             data, _ = raw[channel, :]
#             data = torch.tensor(data, dtype=torch.float32).squeeze()
#             ecogs[channel] = data 
#         return data

# X = load_brainvision_file("data\sub-000_ses-right_task-force_run-0_ieeg.vhdr").reshape(-1, 1)

# def apply_tsne_and_plot_raw(X, n_components=1, perplexity=30, random_state=0): #n dimensions, 
#     """
#     Applies t-SNE to raw data and creates a scatter plot.

#     Parameters:
#     X : array-like, shape (n_samples, n_features)
#         The data to be reduced.
#     n_components : int, default=2
#         Number of dimensions for t-SNE (2 or 3 for visualization).
#     perplexity : float, default=30
#         Balances local vs. global structure. Larger datasets may need higher values.
#     random_state : int, default=0
#         Seed for reproducibility.

#     Returns:
#     None
#     """
#     tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
#     X_tsne = tsne.fit_transform(X)

#     # Plot in 2D
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=10)
#     plt.title("t-SNE Visualization of Raw Data")
#     plt.xlabel("t-SNE Component 1")
#     plt.ylabel("t-SNE Component 2")
#     plt.show()

# apply_tsne_and_plot_raw(X)  # Where X is your raw high-dimensional dataset


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

import numpy as np
import pandas as pd
import mne
import pickle
from multiprocessing import cpu_count


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())

class MyNewDataClass(BaseData):
    """
    Dataset class for welding dataset.
    Attributes:
        all_df: DataFrame indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_dict: Dictionary where keys are IDs, and values are DataFrames for each sample.
        feature_names: Names of columns contained in `feature_dict` (same as feature_df.columns in the previous version).
        all_IDs: IDs contained in the dataset (same as `all_df.index.unique()`).
        max_seq_len: Maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, the script argument overrides this attribute).
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        self.set_num_processes(n_proc=n_proc)

        self.max_seq_len = 250
        self.all_df = self.load_npy()
        self.all_df = self.all_df.set_index('ID')
        self.all_IDs = self.all_df.index.unique()

        # Limit dataset size if specified
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # Interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = list(self.all_df.columns)
        self.feature_df = self.all_df[self.feature_names]

        self.feature_dict = self.to_dict()

    def to_dict(self):
        """
        Converts the DataFrame to a dictionary indexed by IDs.
        Returns:
            dict: A dictionary where keys are IDs, and values are corresponding time series DataFrames.
        """
        feature_dict = {}
        for ID, group in self.all_df.groupby(level=0):
            feature_dict[ID] = group.reset_index(drop=True)
        return feature_dict

    def save_preprocessed_dict(self, file_path="preprocessed_data.pkl"):
        """
        Saves the preprocessed dictionary as a pickle file.
        Args:
            file_path: Path to save the dictionary.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.feature_dict, f)
        # logger.info(f"Preprocessed data dictionary saved to {file_path}.")

    def load_all_mockup_random(self):
        """
        Mockup method to generate random data for testing.
        Returns:
            all_df: A random DataFrame with sequences and IDs.
        """
        N_ROWS = self.max_seq_len * 300
        N_COLS = 5
        all_df = pd.DataFrame(np.random.rand(N_ROWS, N_COLS), columns=[f"col_{i}" for i in range(N_COLS)])
        all_df['ID'] = np.repeat(np.arange(1, N_ROWS // self.max_seq_len + 1), self.max_seq_len)
        return all_df

    def load_all_bv(self):
        """
        Loads and preprocesses BrainVision data.
        Returns:
            all_df: A DataFrame with BrainVision data, standardized per ID.
        """
        PATH_ = "/Users/Timon/Documents/Data/BIDS_BERLIN/sub-002/ses-EphysMedOff01/ieeg/sub-002_ses-EphysMedOff01_task-Rest_acq-StimOff_run-01_ieeg.vhdr"
        raw = mne.io.read_raw_brainvision(PATH_)
        raw.resample(250)
        raw.filter(0.5, 90)
        raw.pick([ch for ch in raw.ch_names if "ECOG" in ch])
        data = raw.get_data()
        all_df = pd.DataFrame(data.T, columns=raw.ch_names)

        ID_col = np.repeat(np.arange(1, all_df.shape[0] // self.max_seq_len + 1), self.max_seq_len)
        all_df = all_df.iloc[:len(ID_col)]
        all_df['ID'] = ID_col

        columns_to_standardize = all_df.columns[:-1]
        all_df[columns_to_standardize] = all_df.groupby('ID')[columns_to_standardize].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        return all_df

    def load_npy(self):
        CLUSTER = True
        if CLUSTER:
            data = np.load("npy_data/all_subs.npy").astype(np.float64)

            total_samples = data.size
            target_size = 4 * self.max_seq_len 
            if total_samples % target_size != 0:
                raise ValueError(
                    f"Cannot reshape array of size {total_samples} into (ID, 4, {self.max_seq_len}). "
                    f"Size must be divisible by {target_size}."
                )

            num_ids = total_samples // target_size
            reshaped_data = data.reshape(num_ids, 4, self.max_seq_len)

            all_df = pd.DataFrame(
                reshaped_data.reshape(-1, 4),
                columns=[f"channel_{i}" for i in range(4)]
            )
            all_df['ID'] = np.repeat(np.arange(1, num_ids + 1), self.max_seq_len)
            columns_to_standardize = all_df.columns[:-1]

            all_df[columns_to_standardize] = all_df.groupby('ID')[columns_to_standardize].transform(
                lambda x: np.clip((x - x.mean()) / x.std(), -9, 9)
            )
        else:
            all_df = pd.read_csv("rcs02l_standardized.csv")

        return all_df


p = MyNewDataClass(root_dir="npy_data")
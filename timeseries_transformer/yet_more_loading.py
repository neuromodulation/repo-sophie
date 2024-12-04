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
from torch.utils.data import Dataset
import os

class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())

class npy_pre_save(BaseData, Dataset):
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

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None, preprocessed="preprocessed.pkl"):
        super().__init__()
        self.set_num_processes(n_proc=n_proc)

        self.max_seq_len = 250
        self.preprocessed = preprocessed
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, 'rb') as f:
                self.feature_dict = pickle.load(f)
            print(f"Loaded preprocessed data from {self.preprocessed}")
        else:
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


            self.feature_names = [col for col in self.all_df.columns if col.startswith("channel_")]
            self.save_preprocessed_dict()
       
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

    def save_preprocessed_dict(self):
        """
        Saves the preprocessed dictionary as a pickle file.
        Args:
            file_path: Path to save the dictionary.
        """
        with open(self.preprocessed, 'wb') as f:
            pickle.dump(self.all_df, f)
        print(f"saved to {self.preprocessed}")    
        # logger.info(f"Preprocessed data dictionary saved to {file_path}.")

    def __len__(self):
        return len(self.feature_dict)    
    
    def __getitem__(self, idx):
        ID = list(self.feature_dict.keys())[idx]
        return self.feature_dict[ID], ID
    
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
        # all_df["feature_df"] = pd.concat(self.data.values(), ignore_index=True)

        ID_col = np.repeat(np.arange(1, all_df.shape[0] // self.max_seq_len + 1), self.max_seq_len)
        all_df = all_df.iloc[:len(ID_col)]
        all_df['ID'] = ID_col

        columns_to_standardize = all_df.columns[:-1]
        all_df[columns_to_standardize] = all_df.groupby('ID')[columns_to_standardize].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        return all_df

    def load_npy(self):
        data = np.load("npy_data/sub_rcs02l.npy").astype(np.float64)

        total_samples = data.size
        num_elements_per_sample = 4 * self.max_seq_len 

        if total_samples % num_elements_per_sample != 0:
            valid_size = (total_samples // num_elements_per_sample) * num_elements_per_sample
            print(f"Clipping data from size {total_samples} to {valid_size} to fit 1s segments.")
            data = data[:valid_size]

        num_ids = total_samples // num_elements_per_sample
        reshaped_data = data.reshape(num_ids, 4, self.max_seq_len)

        all_df = pd.DataFrame(
            reshaped_data.reshape(-1, 4),
            columns=[f"channel_{i}" for i in range(4)]
        )
        all_df["feature_df"] = pd.concat(all_df, ignore_index=True)
        all_df['ID'] = np.repeat(np.arange(1, num_ids + 1), self.max_seq_len)
        columns_to_standardize = all_df.columns[:-1]

        all_df[columns_to_standardize] = all_df.groupby('ID')[columns_to_standardize].transform(
            lambda x: np.clip((x - x.mean()) / x.std(), -9, 9)
        )
   
        return all_df #shape (16210000, 5) with sub_rcs02l.npy: checks out with num_ids=64840*250; 5 columns ['channel_0', 'channel_1', 'channel_2', 'channel_3', 'ID'], (all_df._data.shape)=(5, 16210000)


p = npy_pre_save(root_dir="npy_data")
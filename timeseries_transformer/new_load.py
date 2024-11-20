import os
import glob
import torch
import pandas as pd
import mne
import numpy as np
import re
from datasets.data import BaseData
from torch.utils.data import Dataset


class TSRegressionArchive:
    """
    Dataset class for datasets included in:
    1) the Time Series Regression Archive (www.timeseriesregression.org), or
    2) the Time Series Classification Archive (www.timeseriesclassification.com)
    
    Attributes:
        all_df: (num_samples * seq_len, num_columns) DataFrame indexed by integer indices, with multiple rows
                corresponding to the same index (sample). Each row is a time step; Each column contains either
                metadata (e.g., timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) DataFrame; contains the subset of columns of `all_df`
                    which correspond to selected features.
        feature_names: Names of columns contained in `feature_df` (same as feature_df.columns).
        all_IDs: (num_samples,) Series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique()).
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample.
        max_seq_len: Maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
                     (Moreover, script argument overrides this attribute).
    """
    
    def __init__(self, root_dir, file_list=None, pattern=None, limit_size=None, config=None, preload=True):
        """
        Args:
            root_dir: Directory containing `.vhdr` files.
            file_list: List of specific files to load. Defaults to all files in `root_dir`.
            pattern: Regex pattern to filter files. Defaults to None.
            limit_size: Limit the number of samples. Can be an integer or a proportion (0, 1].
            config: Configuration dictionary for tasks (regression, classification, etc.).
            preload: Whether to preload data using MNE.
        """
        self.root_dir = root_dir
        self.file_list = file_list
        self.pattern = pattern
        self.limit_size = limit_size
        self.config = config or {}
        self.preload = preload

        self.all_files_data = self.load_all_files()
        self.all_df = self.get_combined_data()
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.get_feature_names()
        self.feature_df = self.all_df[self.feature_names]
        self.labels_df = self.initialize_labels()
        self.max_seq_len = self.config.get("max_seq_len", None)

    def load_all_files(self):
        """
        Load all `.vhdr` files in the specified directory and process them.
        Returns:
            A dictionary where keys are file names and values are DataFrames.
        """
        file_paths = (
            glob.glob(os.path.join(self.root_dir, "*.vhdr"))
            if self.file_list is None
            else [os.path.join(self.root_dir, file) for file in self.file_list]
        )
        if self.pattern:
            file_paths = [f for f in file_paths if re.search(self.pattern, f)]
        if not file_paths:
            raise ValueError(f"No `.vhdr` files found in directory: {self.root_dir}")

        all_files_data = {}
        for filepath in file_paths:
            file_name = os.path.basename(filepath)
            df = self.load_single_file(filepath)
            all_files_data[file_name] = df

        return all_files_data

    def load_single_file(self, filepath):
        """
        Load a single `.vhdr` file and process its channels and timestamps.
        Args:
            filepath: Path to the `.vhdr` file.
        Returns:
            A DataFrame where each row represents a channel, and columns represent time-series data.
        """

        raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)
        timestamps = raw.times 
        available_channels = raw.ch_names

        ecog_data = {}
        for channel in available_channels:
            if "ECOG" not in channel and "LFP" not in channel:
                continue
            data, _ = raw[channel, :]
            ecog_data[channel] = data[0] 

        df = pd.DataFrame(ecog_data)
        df.insert(0, "Timestamp", timestamps)

        file_ID = os.path.basename(filepath).split(".")[0]
        df["FileID"] = file_ID

        return df

    def get_combined_data(self):
        """
        Combine all loaded file DataFrames into a single DataFrame.
        Returns:
            A concatenated DataFrame of all files, indexed by FileID and row number.
        """
        combined_df = pd.concat(
            [
                df for file_name, df in self.all_files_data.items()
            ],
            ignore_index=True,
        )
        return combined_df

    def get_feature_names(self):
        """
        Get the feature column names from the combined data.
        Returns:
            A list of feature names.
        """
        return [col for col in self.all_df.columns if col != "Timestamp"]

    def initialize_labels(self):
        """
        Initialize the labels DataFrame. Placeholder implementation (can be updated for specific tasks).
        Returns:
            An empty DataFrame (default) or task-specific labels.
        """
        if self.config.get("task") == "regression":
            return pd.DataFrame(index=self.all_IDs, columns=["Label"])
        elif self.config.get("task") == "classification":
            return pd.DataFrame(index=self.all_IDs, columns=["Class"])
        return pd.DataFrame(index=self.all_IDs) #imputation
    

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask
    
def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask    

class ImputationDataset(Dataset):
    """
    Dataset class to handle raw BIDS ECoG/LFP data. Dynamically preprocesses each channel
    and computes missingness (noise) masks for imputation.
    """

    def __init__(self, root_dir, file_list=None, pattern=None, mean_mask_length=3, masking_ratio=0.15,
                 mode='separate', distribution='geometric', exclude_feats=None, preload=True):
        """
        Args:
            root_dir: Directory containing `.vhdr` files.
            file_list: Optional list of specific files to load.
            pattern: Optional regex pattern to filter files.
            mean_mask_length: Average length of masked segments.
            masking_ratio: Fraction of data to mask.
            mode: Masking mode ('separate' or 'block').
            distribution: Mask length distribution ('geometric' or other).
            exclude_feats: List of features/channels to exclude from masking.
            preload: Whether to preload data with MNE's `read_raw_brainvision`.
        """
        super(ImputationDataset, self).__init__()

        self.root_dir = root_dir
        self.file_list = file_list
        self.pattern = pattern
        self.mean_mask_length = mean_mask_length
        self.masking_ratio = masking_ratio
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats
        self.preload = preload

        # Load and preprocess data
        self.data = self.load_all_files()
        # self.IDs = list(self.data.keys())  # File IDs
        self.all_IDs = list(self.data.keys()) 
        self.feature_df = pd.concat(self.data.values(), ignore_index=True)

    def load_all_files(self):
        """
        Load all `.vhdr` files in the directory, preprocess channels, and return as DataFrames.
        Returns:
            A dictionary where keys are file names and values are preprocessed DataFrames.
        """
        file_paths = (
            glob.glob(os.path.join(self.root_dir, "*.vhdr"))
            if self.file_list is None
            else [os.path.join(self.root_dir, file) for file in self.file_list]
        )

        if self.pattern:
            file_paths = [f for f in file_paths if re.search(self.pattern, f)]
        if not file_paths:
            raise ValueError(f"No `.vhdr` files found in directory: {self.root_dir}")

        all_files_data = {}
        for filepath in file_paths:
            file_name = os.path.basename(filepath)
            df = self.load_single_file(filepath)
            all_files_data[file_name] = df

        return all_files_data

    def load_single_file(self, filepath):
        """
        Load and preprocess a single `.vhdr` file.
        Args:
            filepath: Path to the `.vhdr` file.
        Returns:
            A DataFrame where each row represents a channel's time series data, along with timestamps.
        """
        raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)
        timestamps = raw.times
        available_channels = raw.ch_names

        ecog_data = {}
        for channel in available_channels:
            if "ECOG" not in channel and "LFP" not in channel:
                continue
            data, _ = raw[channel, :]
            ecog_data[channel] = data[0]

        df = pd.DataFrame(ecog_data)
        df.insert(0, "Timestamp", timestamps)

        file_ID = os.path.basename(filepath).split(".")[0]
        df["FileID"] = file_ID

        return df

    def __getitem__(self, ind):
        """
        For a given integer index, preprocess the corresponding sample (channel data).
        Args:
            ind: Integer index of sample in dataset.
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample.
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s unaffected input.
            ID: ID of the file or sample.
        """
        ID = self.IDs[ind]
        sample_df = self.feature_df[self.feature_df["FileID"] == ID]

        # Drop non-feature columns (e.g., Timestamp, FileID)
        X = sample_df.drop(columns=["Timestamp", "FileID"]).values  # (seq_length, feat_dim)
        mask = noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
                          self.exclude_feats)  # (seq_length, feat_dim)

        return torch.from_numpy(X), torch.from_numpy(mask), ID

    def update(self):
        """
        Update masking parameters to progressively increase difficulty.
        """
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        """
        Returns the total number of unique samples in the dataset.
        """
        return len(self.IDs)


cooler_data = {"bids": ImputationDataset}
               
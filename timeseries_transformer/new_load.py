import os
import glob
import torch
import pandas as pd
import mne
import numpy as np
import re
from datasets.data import BaseData

# class TSRegressionArchive(BaseData):
#     """
#     Dataset class for datasets included in:
#     1) the Time Series Regression Archive (www.timeseriesregression.org), or
#     2) the Time Series Classification Archive (www.timeseriesclassification.com)
    
#     Updates:
#         - Supports `.vhdr` files in BIDS format.
#         - Separates channels into individual rows/columns.
#         - Each file becomes its own DataFrame.
#     """
#     def __init__(self, root_dir, file_list=None, pattern=None, limit_size=None, config=None, preload=True):
#         """
#         Args:
#             root_dir: Directory containing `.vhdr` files.
#             file_list: List of specific files to load. Defaults to all files in `root_dir`.
#             pattern: Regex pattern to filter files. Defaults to None.
#             limit_size: Limit the number of samples. Can be an integer or a proportion (0, 1].
#             config: Configuration dictionary for tasks (regression, classification, etc.).
#             preload: Whether to preload data using MNE.
#         """
#         self.root_dir = root_dir
#         self.file_list = file_list
#         self.pattern = pattern
#         self.limit_size = limit_size
#         self.config = config or {}
#         self.preload = preload

#         self.all_files_data = self.load_all_files()
#         self.labels_df = None

#     def load_all_files(self):
#         file_paths = (
#             glob.glob(os.path.join(self.root_dir, "*.vhdr"))
#             if self.file_list is None
#             else [os.path.join(self.root_dir, file) for file in self.file_list]
#         )
#         if self.pattern:
#             file_paths = [f for f in file_paths if re.search(self.pattern, f)]
#         if not file_paths:
#             raise ValueError(f"No `.vhdr` files found in directory: {self.root_dir}")

#         all_files_data = {}
#         for filepath in file_paths:
#             file_name = os.path.basename(filepath)
#             df = self.load_single_file(filepath)
#             all_files_data[file_name] = df

#         return all_files_data

#     def load_single_file(self, filepath):
#         """
#         Load a single `.vhdr` file and process its channels and timestamps.
#         Args:
#             filepath: Path to the `.vhdr` file.
#         Returns:
#             A DataFrame where each row represents a channel, and columns represent time-series data.
#         """
#         raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)
#         timestamps = raw.times 

#         available_channels = raw.ch_names
#         ecog_data = {}
#         for channel in available_channels:
#             if "ECOG" not in channel and "LFP" not in channel:
#                 continue
#             data, _ = raw[channel, :]
#             ecog_data[channel] = data[0]

#         df = pd.DataFrame(ecog_data)
#         df.insert(0, "Timestamp", timestamps) 

#         return df

#     def get_all_data(self):
#         """
#         Combine all loaded files into a single DataFrame for further analysis.
#         Returns:
#             A concatenated DataFrame of all files, with a new column indicating the file source.
#         """
#         combined_df = pd.concat(
#             [
#                 df.assign(FileName=file_name)
#                 for file_name, df in self.all_files_data.items()
#             ],
#             ignore_index=True,
#         )
#         return combined_df

# cooler_data = {"bids": TSRegressionArchive}

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

cooler_data = {"bids": TSRegressionArchive}
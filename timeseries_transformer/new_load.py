import os
import glob
import torch
import pandas as pd
import mne
import numpy as np
import re
from datasets.data import BaseData

class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
    1) the Time Series Regression Archive (www.timeseriesregression.org), or
    2) the Time Series Classification Archive (www.timeseriesclassification.com)
    
    Updates:
        - Supports `.vhdr` files in BIDS format.
        - Separates channels into individual rows/columns.
        - Each file becomes its own DataFrame.
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
        self.labels_df = None

    def load_all_files(self):
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

        return df

    def get_all_data(self):
        """
        Combine all loaded files into a single DataFrame for further analysis.
        Returns:
            A concatenated DataFrame of all files, with a new column indicating the file source.
        """
        combined_df = pd.concat(
            [
                df.assign(FileName=file_name)
                for file_name, df in self.all_files_data.items()
            ],
            ignore_index=True,
        )
        return combined_df

cooler_data = {"bids": TSRegressionArchive}
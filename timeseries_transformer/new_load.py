import os
import glob
import mne
import pandas as pd
import numpy as np
from aeon.datasets import write_to_tsfile

def subsample(series, factor=1):
    """Subsample the time-series data by a specified factor."""
    if factor > 1:
        return series[::factor]
    return series


def interpolate_missing(series):
    """Interpolate missing values in the time-series."""
    return series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')


def convert_to_ts(filepath, output_tsfile):
    """
    Converts a single BIDS `.vhdr` file to `.ts` format.
    
    Args:
        filepath (str): Path to the BIDS `.vhdr` file.
        output_tsfile (str): Output path for the `.ts` file.
    """
    raw = mne.io.read_raw_brainvision(filepath, preload=True)
    data, times = raw.get_data(return_times=True)

    df = pd.DataFrame(data.T, columns=raw.ch_names, index=pd.Index(times, name='timestamp'))

    lengths = df.applymap(len).values if isinstance(df.iloc[0, 0], pd.Series) else None
    if lengths is not None:
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
        if np.sum(horiz_diffs) > 0:
            print(f"Inconsistent lengths in {filepath}. Subsampling first dimension...")
            df = df.applymap(lambda x: subsample(x, factor=1)) 

    subsample_factor = 1
    df = df.applymap(lambda x: subsample(x, factor=subsample_factor))

    df = df.groupby(df.index).transform(interpolate_missing)

    write_to_tsfile(
        data=df, #needs pd dataframe as input what a shit
        path=output_tsfile,
        problem_name=os.path.basename(filepath).split('.')[0],
        class_labels=None, 
        fold=None,
        return_separate_X_and_y=False,
    )

    print(f"Converted {filepath} to {output_tsfile}")


def batch_convert_bids_to_ts(input_dir, output_dir):
    """
    Converts all `.vhdr` files in a directory to `.ts` format.
    
    Args:
        input_dir (str): Directory containing `.vhdr` files.
        output_dir (str): Directory to save `.ts` files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vhdr_files = glob.glob(os.path.join(input_dir, "*.vhdr"))
    if not vhdr_files:
        print("No .vhdr files found in this directory.")
        return

    for vhdr_file in vhdr_files:
        ts_filename = os.path.splitext(os.path.basename(vhdr_file))[0] + ".ts"
        output_tsfile = os.path.join(output_dir, ts_filename)

        convert_to_ts(vhdr_file, output_tsfile)
    
    print(f"All files converted. Outputs saved in {output_dir}")


if __name__ == "__main__":
    input_directory = "data"
    output_directory = "ts_data"

    batch_convert_bids_to_ts(input_directory, output_directory)


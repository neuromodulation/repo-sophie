# import mne
# import torch
# from pathlib import Path


# def load_data(
#     filepath: str,
#     channel_names: list,
#     target_name: str,
#     window_size: float = 2.0,
#     overlap: float = 0.0,
#     preload: bool = True
# ):

#     raw = mne.io.read_raw_brainvision(filepath, preload=preload)
#     ecogs = []
    
#     for channel in channel_names:
#         data, times = raw[channel, :]
    
#         ecogs.append(torch.tensor(data, dtype=torch.float32).squeeze())

#     target, time = raw[target_name, :]

#     x_data = torch.stack(ecogs, dim=1).unsqueeze(0)
#     x_data = (x_data - x_data.mean()) / x_data.std()
#     x_data = x_data.permute(0, 2, 1)

#     y_data = torch.tensor(target.T, dtype=torch.float32).view(1, 1, -1)

#     def create_sliding_windows(data, window_size, overlap, sfreq):
#         step = int(window_size * sfreq)
#         overlap_step = int(overlap * sfreq)
#         data_length = data.shape[2]
#         windows = []

#         for start in range(0, data_length - step + 1, step - overlap_step):
#             stop = start + step
#             windows.append(data[:, :, start:stop])
        
#         return windows

#     sfreq = raw.info['sfreq']

#     x_windows = create_sliding_windows(x_data, window_size, overlap, sfreq)
#     y_windows = create_sliding_windows(y_data, window_size, overlap, sfreq)

#     return x_windows, y_windows

# channel_names = ['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5']
# target_name = 'MOV_LEFT_CLEAN'

# x_windows, y_windows = load_data(
#     filepath="data/sub-000_ses-right_task-force_run-0_ieeg.vhdr",
#     channel_names=channel_names,
#     target_name=target_name,
#     window_size=2.0,
#     overlap=0.0
# )

# print(f"number of X-Windows: {len(x_windows)}")
# print(f"size of X-Window: {x_windows[0].shape}")
# print(f"size of Y-Window: {y_windows[0].shape}")
# print(x_windows[0])
# print(y_windows[0])


#datashape in data
import mne
import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy

class BIDSBrainVisionDataset(Dataset):
    def __init__(self, directory, channel_names, target_name, window_size=2.0, overlap=0.0, preload=True): #preload=False for goin easy on the RAM
        self.directory = Path(directory)
        self.channel_names = channel_names
        self.target_name = target_name
        self.window_size = window_size
        self.overlap = overlap
        self.preload = preload
        
        self.filepaths = list(self.directory.glob("*.vhdr"))
        
        self.windows = []
        self._prepare_dataset()

    def _load_brainvision_file(self, filepath):
        raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)
        ecogs = []

        for channel in self.channel_names:
            data, _ = raw[channel, :]
            ecogs.append(torch.tensor(data, dtype=torch.float32))

        target, _ = raw[self.target_name, :]

        x_data = torch.stack(ecogs, dim=1).unsqueeze(0)  ####maybe worng stack (after permuting?)
        x_data = (x_data - x_data.mean()) / x_data.std()
        x_data = x_data.squeeze(1)
        # x_data = x_data.permute(0, 2, 1)
        # x_data = x_data.mean(dim=1, keepdim=True) ####################################################

        y_data = torch.tensor(target.T, dtype=torch.float32).reshape(1, 1, -1)
#ydata: torch.Size([1, 1, 130001]), xdata: torch.Size([1, 6, 130001])
        return x_data, y_data, raw.info['sfreq']
    
    def _sliding_windows(self, data, window_size, overlap, sfreq):
        step = int(window_size * sfreq)
        overlap_step = int(overlap * sfreq)
        data_length = data.shape[2]
        windows = []

        for x in range(0, data_length - step + 1, step - overlap_step):
            stop = x + step
            print(f"window from {x} to {stop}")
            windows.append(data[:, :, x:stop])
            # print(f"window size: {window_size}")
        print(f"number of windows={len(windows)}")
        return windows
    
    def _prepare_dataset(self):
        for filepath in self.filepaths:
            print(f"loading file: {filepath}")
            x_data, y_data, sfreq = self._load_brainvision_file(filepath)
            
            x_windows = self._sliding_windows(x_data, self.window_size, self.overlap, sfreq)
            y_windows = self._sliding_windows(y_data, self.window_size, self.overlap, sfreq)
            
            for x_window, y_window in zip(x_windows, y_windows):
                self.windows.append((x_window, y_window))
            print(f"len(windows)={len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

channel_names = ['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5']
target_name = 'MOV_LEFT_CLEAN'

dataset = BIDSBrainVisionDataset(
    directory="data",
    channel_names=channel_names,
    target_name=target_name,
    window_size=2.0,
    overlap=0.0
)

for i in range(3):
    x, y = dataset[i]
    print(f"window {i}: x-data {x.shape}, y-data {y.shape}")

import mne
import torch
from pathlib import Path


def load_data(
    filepath: str,
    channel_names: list,
    target_name: str,
    window_size: float = 2.0,
    overlap: float = 0.0,
    preload: bool = True
):

    raw = mne.io.read_raw_brainvision(filepath, preload=preload)
    ecogs = []
    
    for channel in channel_names:
        data, times = raw[channel, :]
    
        ecogs.append(torch.tensor(data, dtype=torch.float32).squeeze())

    target, time = raw[target_name, :]

    x_data = torch.stack(ecogs, dim=1).unsqueeze(0)
    x_data = (x_data - x_data.mean()) / x_data.std()
    x_data = x_data.permute(0, 2, 1)

    y_data = torch.tensor(target.T, dtype=torch.float32).view(1, 1, -1)

    def create_sliding_windows(data, window_size, overlap, sfreq):
        step = int(window_size * sfreq)
        overlap_step = int(overlap * sfreq)
        data_length = data.shape[2]
        windows = []

        for start in range(0, data_length - step + 1, step - overlap_step):
            stop = start + step
            windows.append(data[:, :, start:stop])
        
        return windows

    sfreq = raw.info['sfreq']

    x_windows = create_sliding_windows(x_data, window_size, overlap, sfreq)
    y_windows = create_sliding_windows(y_data, window_size, overlap, sfreq)

    return x_windows, y_windows

channel_names = ['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5']
target_name = 'MOV_LEFT_CLEAN'

x_windows, y_windows = load_data(
    filepath="sub-000_ses-right_task-force_run-0_ieeg.vhdr",
    channel_names=channel_names,
    target_name=target_name,
    window_size=2.0,
    overlap=0.0
)

print(f"number of X-Windows: {len(x_windows)}")
print(f"size of X-Window: {x_windows[0].shape}")
print(f"size of Y-Window: {y_windows[0].shape}")
print(x_windows[0])


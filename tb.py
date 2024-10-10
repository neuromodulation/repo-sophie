# import tensorflow as tf

# for e in tf.train.summary_iterator("logging_events\events.out.tfevents.1728032688.DESKTOP-RPLNF8J.6044.0"):
#     for v in e.summary.value:
#         if v.tag == 'loss' or v.tag == 'accuracy':
#             print(v.simple_value)
import torch
from pathlib import Path
import mne
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

class BIDSBrainVisionDataset(Dataset):
    def __init__(self, directory, channel_names, target_name, window_size=2.0, overlap=0.0, preload=True, feature_extractor=None):
        self.directory = Path(directory)
        self.channel_names = channel_names
        self.target_name = target_name
        self.window_size = window_size
        self.overlap = overlap
        self.preload = preload
        self.feature_extractor = feature_extractor
        self.data = []
        self.filepaths = list(self.directory.glob("*.vhdr"))
        self._load_all()
        
        self.windows = []
        # self._prepare_dataset()

    def _load_brainvision_file(self, filepath):
        raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)

        available_channels = [ch_name for ch_name in self.channel_names if ch_name in raw.ch_names]
        ecogs = {}

        for channel in available_channels:
            data, _ = raw[channel, :]  
            ecogs[channel] = torch.tensor(data, dtype=torch.float32) 
        target, _ = raw[self.target_name, :]
        y_data = torch.tensor(target.T, dtype=torch.float32).reshape(1, -1)
        
        return ecogs, y_data, raw.info['sfreq'], available_channels #ecogs: dict, ecogs[0]: tensor
    
    def _load_all(self):
        for filepath in self.filepaths:
            ecog_channels, y_data, sfreq, available_channels = self._load_brainvision_file(filepath)

        for channel_name, channel_data in ecog_channels.items():
            self.data.append((channel_data.unsqueeze(0), y_data.unsqueeze(0)))

    # def _sliding_windows(self, data, window_size, overlap, sfreq):
    #     step = int(window_size * sfreq)
    #     overlap_step = int(overlap * sfreq)
    #     data_length = data.shape[0]
    #     windows = []

    #     for x in range(0, data_length - step + 1, step - overlap_step):
    #         stop = x + step
    #         windows.append(data[x:stop])
    #     return windows #list

    # def _prepare_dataset(self):
    #     for filepath in self.filepaths:
    #         ecog_channels, y_data, sfreq, available_channels = self._load_brainvision_file(filepath)
            
    #         for channel_name, channel_data in ecog_channels.items():
    #             x_windows = self._sliding_windows(channel_data, self.window_size, self.overlap, sfreq) #list
    #             y_windows = self._sliding_windows(y_data.squeeze(0), self.window_size, self.overlap, sfreq)

    #             for x_window, y_window in zip(x_windows, y_windows):
    #                 self.windows.append((x_window.unsqueeze(0), y_window.unsqueeze(0)))  

    # def __len__(self):
    #     return len(self.windows)

    # def __getitem__(self, idx):
    #     x_windows, y_windows = self.windows[idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_data, y_data = self.data[idx]
        x_data = x_data.squeeze(0)
        y_data = y_data.squeeze()

        if self.feature_extractor:
            inputs = self.feature_extractor(x_data.numpy(), sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt")
            x_data = inputs["input_values"][0]
        return x_data, y_data
    #     if self.feature_extractor:
    #         inputs = self.feature_extractor(x_window.numpy(), sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt")
    #         x_window = inputs["input_values"][0]
    #     return x_window, y_window

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")


dataset = BIDSBrainVisionDataset(
    directory="data",
    channel_names=['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5'],  # List all potential channels
    target_name='MOV_LEFT_CLEAN',
    window_size=2.0,
    overlap=0.0,
    feature_extractor=feature_extractor
)

for x in range(min(3, len(dataset))):
    x, y = dataset[x]
    print(f"sample{x}, x_data shape: {x.shape}, y_data shape: {y.shape}")

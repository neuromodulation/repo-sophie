import torch
from pathlib import Path
import mne
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from datasets import Dataset as HFDataset, Audio
import numpy as np

class BIDSBrainVisionDataset(Dataset):
    def __init__(self, directory, output_dir, preload=True, feature_extractor=None, target_sr=16000, debugging_mode=False):
        self.directory = Path(directory)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preload = preload
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.data = []
        self.debugging_mode = debugging_mode

        self.filepaths = list(self.directory.glob("*.vhdr"))
        if len(self.filepaths) == 0:
            raise ValueError(f"No .vhdr files found in the directory: {directory}")
        
        if self.debugging_mode:
            self.filepaths = self.filepaths[:2]
            print("debugging mdoe is ON, loading only firat two files")
        else:
            print("debugging mode OFF")
        
        self._load_all_files()
        self.hf_dataset = self._create_hf_dataset()

    def _load_brainvision_file(self, filepath):
        print(f"filepath:  {filepath}")
        raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)
        
        available_channels = raw.ch_names
        ecogs = {}

        for channel in available_channels:
            if "ECOG" not in channel and "LFP" not in channel: #didnt find, where the channeltypes get read with raw, so i just filter by name
                continue
            data, _ = raw[channel, :]
            ecogs[channel] = torch.tensor(data, dtype=torch.float32)

        return ecogs, raw.info['sfreq'], list(ecogs.keys())

    def _load_all_files(self):
        for filepath in self.filepaths:
            ecog_channels, sfreq, available_channels = self._load_brainvision_file(filepath)

            for channel_name, channel_data in ecog_channels.items():
                flac_filepath = self._save_channel_as_flac(channel_data, channel_name, sfreq, filepath.stem)
                self.data.append((flac_filepath, None))
            print(f"Processed and saved all channels for file: {filepath}")
        print(f"Total number of samples added to dataset: {len(self.data)}")

    def _save_channel_as_flac(self, channel_data, channel_name, sfreq, file_stem):
        flac_filename = self.output_dir / f"{file_stem}_{channel_name}.flac"
        if flac_filename.exists():
            return flac_filename
        
        channel_data = channel_data.numpy().astype(np.float32).squeeze()
        channel_data_resampled = librosa.resample(channel_data, orig_sr=sfreq, target_sr=self.target_sr) #Bandlimited sinc Interpolation
        channel_data_resampled = librosa.util.normalize(channel_data_resampled)
        sf.write(flac_filename, channel_data_resampled, self.target_sr, format='FLAC')

        print(f"Saved FLAC for {channel_name} as {flac_filename}")
        return flac_filename

    def _create_hf_dataset(self):
        audio_files = [str(entry[0]) for entry in self.data]  
        transcriptions = [None] * len(self.data)

        data_dict = {
            "audio": audio_files,
            "transcription": transcriptions
        }

        hf_dataset = HFDataset.from_dict(data_dict)
        hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=self.target_sr))

        return hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base-v2")
# train_dataset = BIDSBrainVisionDataset(
#     directory="data",
#     output_dir="output_flac",
#     feature_extractor=feature_extractor,
#     target_sr=16000,
#     debugging_mode=True
# )
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



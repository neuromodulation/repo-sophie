import torch
import os

import torch
from pathlib import Path
import mne
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from datasets import Dataset as HFDataset, Audio
import numpy as np
from datasets import DatasetDict, concatenate_datasets

feature_extractor = Wav2Vec2FeatureExtractor

preprocessed_dir = "preprocessed_data"
os.makedirs(preprocessed_dir, exist_ok=True)

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
            self.filepaths = self.filepaths[:1]
            print("debugging mdoe is ON, loading only firat file")
        else:
            print("debugging mode OFF")
        
        self._load_all_files()
        self.hf_dataset = self._create_hf_dataset()

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
        channel_data_resampled = librosa.resample(channel_data, orig_sr=sfreq, target_sr=self.target_sr)
        # channel_data_resampled = librosa.util.normalize(channel_data_resampled)
        sf.write(flac_filename, channel_data_resampled, self.target_sr, format='FLAC')

        print(f"Saved FLAC for {channel_name} as {flac_filename}")
        return flac_filename

    def _create_hf_dataset(self):
        audio_files = [str(entry[0]) for entry in self.data]  
        transcriptions = [None] * len(self.data)

        data_dict = {
            "raw_data": audio_files,
            # "transcription": transcriptions
        }

        hf_dataset = HFDataset.from_dict(data_dict)
        hf_dataset = hf_dataset.cast_column("raw_data", Audio(sampling_rate=self.target_sr))

        return hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]


    datasets_splits = []
    datasets_splits.append(train_dataset.hf_dataset)
     
    raw_datasets = DatasetDict()
    
    num_validation_samples = max(1, raw_datasets["train"].num_rows * 10 // 100)

    if num_validation_samples == 0:
        raise ValueError(
            "`args.validation_split_percentage` is less than a single sample "
            f"for {len(raw_datasets['train'])} training samples. Increase "
            "`args.num_validation_split_percentage`."
        )

    raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
    raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))
########

def load_brainvision_file(self, filepath):
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

def prepare_dataset(batch):
    sample = batch["raw_data"]
    max_length = int(feature_extractor.sampling_rate * 20.0)
    inputs = feature_extractor(
        sample["array"], 
        sampling_rate=sample["sampling_rate"], 
        max_length=max_length, 
        truncation=True
    )
    input_values = inputs.input_values[0] #[320000, ]

    batch["input_values"] = input_values
    batch["input_length"] = len(input_values)

    return batch

for idx, filepath in enumerate(filepaths):
    ecog_channels, sfreq, _ = load_brainvision_file(filepath)

    for channel_name, channel_data in ecog_channels.items():
        windows = sliding_windows(channel_data, sfreq)  
        
        for window_idx, window in enumerate(windows):
            inputs = feature_extractor(
                window, sampling_rate=sfreq, return_tensors="pt"
            ).input_values[0]

            save_path = os.path.join(preprocessed_dir, f"sample_{idx}_{channel_name}_{window_idx}.pt")
            torch.save(inputs, save_path)

# vectorized_datasets: datasetDict, column_names: {"train": ["input_values", "input_length"], "validation": ["input_values", "input_length"]}
# shape=x, 2
# vectorized_datasets = vectorized_datasets.remove_columns("input_length")
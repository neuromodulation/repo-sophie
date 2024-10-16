# import torch
# from pathlib import Path
# import mne
# from torch.utils.data import Dataset
# import librosa
# import soundfile as sf
# from transformers import Wav2Vec2FeatureExtractor
# import numpy as np

# class BIDSBrainVisionDataset(Dataset):
#     def __init__(self, directory, channel_names, target_name, output_dir, preload=True, feature_extractor=None, target_sr=16000):
#         self.directory = Path(directory)
#         self.channel_names = channel_names
#         self.target_name = target_name
#         self.output_dir = Path(output_dir) 
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.preload = preload
#         self.feature_extractor = feature_extractor
#         self.target_sr = target_sr
#         self.data = []

#         self.filepaths = list(self.directory.glob("*.vhdr"))
#         if len(self.filepaths) == 0:
#             raise ValueError(f"No .vhdr files found in the directory: {directory}")
        
#         self._load_all_files()

#     def _load_brainvision_file(self, filepath):
#         raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)

#         available_channels = [ch_name for ch_name in self.channel_names if ch_name in raw.ch_names]
#         ecogs = {}
       
#         for channel in available_channels:
#             data, _ = raw[channel, :]
#             ecogs[channel] = torch.tensor(data, dtype=torch.float32) 

#         target, _ = raw[self.target_name, :]
#         y_data = torch.tensor(target.T, dtype=torch.float32).reshape(1, -1)
        
#         return ecogs, y_data, raw.info['sfreq'], available_channels

#     def _save_channel_as_flac(self, channel_data, channel_name, sfreq, file_idx):
        
#         channel_data = channel_data.numpy().astype(np.float32).squeeze()
#         channel_data_resampled = librosa.resample(channel_data, orig_sr=sfreq, target_sr=self.target_sr)
#         channel_data_resampled = librosa.util.normalize(channel_data_resampled)
#         flac_filename = self.output_dir / f"{file_idx}_{channel_name}.flac"
#         sf.write(flac_filename, channel_data_resampled, self.target_sr, format='FLAC')

#         print(f"Saved FLAC for {channel_name} as {flac_filename}")
#         return flac_filename

#     def _load_all_files(self):
    
#         for file_idx, filepath in enumerate(self.filepaths):
#             ecog_channels, y_data, sfreq, available_channels = self._load_brainvision_file(filepath)

#             for channel_name, channel_data in ecog_channels.items():
#                 flac_filepath = self._save_channel_as_flac(channel_data, channel_name, sfreq, file_idx)
#                 self.data.append((flac_filepath, y_data)) 
#                 print(f"Processed and saved all channels for file: {filepath}")
#         print(f"Total number of samples added to dataset: {len(self.data)}")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         flac_filepath, y_data = self.data[idx]
#         audio_signal, _ = librosa.load(flac_filepath, sr=self.target_sr)
#         if self.feature_extractor:
#             inputs = self.feature_extractor(audio_signal, sampling_rate=self.target_sr, return_tensors="pt")
#             audio_signal = inputs["input_values"][0]
#         return audio_signal, y_data


# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# dataset = BIDSBrainVisionDataset(
#     directory="data", 
#     channel_names=['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5'],
#     target_name='MOV_LEFT_CLEAN',
#     output_dir="output_flac",
#     feature_extractor=feature_extractor,
#     target_sr=16000
# )

# audio_signal, y_data = dataset[0]
# print(f"Audio Signal Shape: {audio_signal.shape}, Target Data Shape: {y_data.shape}")

import torch
from pathlib import Path
import mne
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
from datasets import Dataset as HFDataset, Audio

class BIDSBrainVisionDataset(Dataset):
    def __init__(self, directory, channel_names, target_name, output_dir, preload=True, feature_extractor=None, target_sr=16000):
        self.directory = Path(directory)
        self.channel_names = channel_names
        self.target_name = target_name
        self.output_dir = Path(output_dir) 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preload = preload
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.data = []  # List to hold file paths and target data

        # Get the list of BrainVision files (ending in .vhdr)
        self.filepaths = list(self.directory.glob("*.vhdr"))
        if len(self.filepaths) == 0:
            raise ValueError(f"No .vhdr files found in the directory: {directory}")
        
        # Load all files, save channels as FLAC, and prepare the dataset
        self._load_all_files()

        # After generating the FLAC files, we will create a Hugging Face Dataset
        self.hf_dataset = self._create_hf_dataset()

    def _load_brainvision_file(self, filepath):
        raw = mne.io.read_raw_brainvision(filepath, preload=self.preload)

        # Find available channels
        available_channels = [ch_name for ch_name in self.channel_names if ch_name in raw.ch_names]
        ecogs = {}
       
        for channel in available_channels:
            data, _ = raw[channel, :]
            ecogs[channel] = torch.tensor(data, dtype=torch.float32)

        target, _ = raw[self.target_name, :]
        y_data = torch.tensor(target.T, dtype=torch.float32).reshape(1, -1)
        
        return ecogs, y_data, raw.info['sfreq'], available_channels

    def _save_channel_as_flac(self, channel_data, channel_name, sfreq, file_idx):
        # Convert ECoG channel data to numpy and resample
        channel_data = channel_data.numpy().astype(np.float32).squeeze()
        channel_data_resampled = librosa.resample(channel_data, orig_sr=sfreq, target_sr=self.target_sr)
        channel_data_resampled = librosa.util.normalize(channel_data_resampled)

        # Save the resampled audio as FLAC
        flac_filename = self.output_dir / f"{file_idx}_{channel_name}.flac"
        sf.write(flac_filename, channel_data_resampled, self.target_sr, format='FLAC')

        print(f"Saved FLAC for {channel_name} as {flac_filename}")
        return flac_filename

    def _load_all_files(self):
        # Iterate over all BrainVision files, save channels as FLAC, and store paths
        for file_idx, filepath in enumerate(self.filepaths):
            ecog_channels, y_data, sfreq, available_channels = self._load_brainvision_file(filepath)

            for channel_name, channel_data in ecog_channels.items():
                flac_filepath = self._save_channel_as_flac(channel_data, channel_name, sfreq, file_idx)
                self.data.append((flac_filepath, y_data))
            print(f"Processed and saved all channels for file: {filepath}")
        print(f"Total number of samples added to dataset: {len(self.data)}")

    def _create_hf_dataset(self):
        audio_files = [str(entry[0]) for entry in self.data] 
        transcriptions = [entry[1].numpy().tolist() for entry in self.data] 
        
        data_dict = {
            "audio": audio_files,
            "transcription": transcriptions 
        }
      
        hf_dataset = HFDataset.from_dict(data_dict)

        # Cast the audio column to the appropriate feature (Hugging Face automatically decodes the audio)
        hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=self.target_sr))

        return hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

audio_dataset = BIDSBrainVisionDataset(
    directory="data", 
    channel_names=['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5'],
    target_name='MOV_LEFT_CLEAN',
    output_dir="output_flac", 
    feature_extractor=feature_extractor,
    target_sr=16000) 

#sample = dataset[0]
# print(f"Audio Sample: {sample['audio']}, Transcription: {sample['transcription']}")



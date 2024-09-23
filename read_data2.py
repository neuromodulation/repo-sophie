import mne
import torch
from datasets import Dataset

#preload=True for raw data, target_name for target channel

channel_names = ['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5']
target_name = "MOV_LEFT_CLEAN"
vhdr_path = "data\sub-000_ses-right_task-force_run-0_ieeg.vhdr"

class BIDSLoader:
    def __init__(self, vhdr_path, channel_names, target_name, preload=True):

        self.vhdr_path = vhdr_path
        self.channel_names = channel_names
        self.target_name = target_name
        self.preload = preload

    def load_data(self):

        raw = mne.io.read_raw_brainvision(self.vhdr_path, preload=self.preload)
        ecogs = []
        for channel in self.channel_names:
            data, _ = raw[channel, :]
            ecogs.append(torch.tensor(data, dtype=torch.float32).squeeze())
        
        x_data = torch.stack(ecogs, dim=1).unsqueeze(0)
        x_data = (x_data - x_data.mean()) / x_data.std()
        x_data = x_data.permute(0, 2, 1)

        target, _ = raw[self.target_name, :]
        y_data = torch.tensor(target.T, dtype=torch.float32).view(1, 1, -1)

        return x_data, y_data

    def load_dataset(self):

        x_data, y_data = self.load_data()
        data_dict = {
                "input": [x_data], 
                "target": [y_data]}

        return Dataset.from_dict(data_dict)

loader = BIDSLoader(vhdr_path=vhdr_path, channel_names=channel_names, target_name=target_name, preload=True)
x_data, y_data = loader.load_data()
dataset = loader.load_dataset()
print(x_data, y_data, dataset)
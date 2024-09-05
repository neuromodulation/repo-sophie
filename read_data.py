import mne
import torch.nn

#maybe cnn layer, erst auf zeitdimension, dann über räumlich
raw = mne.io.read_raw_brainvision("sub-000_ses-right_task-force_run-0_ieeg.vhdr", preload=True)
#data = raw.get_data()
#raw.pick("ECOG_RIGHT_0")
channel_name = ['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5']

ecogs = []
for p in channel_name:
    data, times = raw[p, :]
    ecogs.append(torch.tensor(data, dtype=torch.float32).squeeze())

target_name = 'MOV_LEFT_CLEAN'
target, time = raw[target_name, :]

#MOV_LEFT_CLEAN contains 2 arrays: first one (target) with only zeros (using that, code runs okay), second one (time) produces only NAN loss after first epoch, dont rlly know what thats about

x_data = torch.stack(ecogs, dim=1).unsqueeze(0) #just testing 4 now
x_data = (x_data - x_data.mean()) / x_data.std()
x_data = x_data.permute(0, 2, 1)
y_data = torch.tensor(time.T, dtype=torch.float32).view(1, 1, 130001) 

print(data)

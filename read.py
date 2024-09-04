import mne
import torch.nn 
import torch.nn.functional as F
import numpy as np

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

x_data = torch.stack(ecogs, dim=1).unsqueeze(0) #just testing 4 now
x_data = (x_data - x_data.mean()) / x_data.std()
x_data = x_data.permute(0, 2, 1)
y_data = torch.tensor(target.T, dtype=torch.float32).view(1, 1, 130001)


class EEGRegression(torch.nn.Module): #ab epoch 13500 keine veränderung mehr, loss 0.057739850133657455; epoch 0, loss 0.2543923556804657
    def __init__(self):
        super(EEGRegression, self).__init__()
        self.linear = torch.nn.Linear(6, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

model1 = EEGRegression()
lossfunc1 = torch.nn.MSELoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001)


class EEGCNNRegr(torch.nn.Module):
    def __init__(self):
        super(EEGCNNRegr, self).__init__()
        self.conv1 = torch.nn.Conv2d(6, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flat = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(16*6500, 100)
        self.act4 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.flat(x)
        x = self.act3(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x
    
model2 = EEGCNNRegr()
lossfunc2 = torch.nn.MSELoss()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    


for i in range(100000): 
    pred_y = model2(x_data)
    loss = lossfunc2(pred_y, y_data)

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    if i % 100 == 0:
        print('epoch {}, loss {}'.format(i, loss.item()))


import torch.nn


class EEGCNNRegr(torch.nn.Module):
    def __init__(self):
        super(EEGCNNRegr, self).__init__()
        self.conv1d = torch.nn.Conv1d(6, 16, kernel_size=3, stride=1)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool1d(2)

        self.conv2d = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flat = torch.nn.Flatten()
        self.fc1 = None
        self.act3 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.act1(self.conv1d(x))
        x = self.pool1(x)

        x = x.unsqueeze(1)
        x = self.act2(self.conv2d(x))
        x = self.pool2(x)

        if self.fc1 is None:
            features = x.view(x.size(0), -1).size(1)
            self.fc1 = torch.nn.Linear(features, 100)

        x = self.flat(x)
        x = self.act3(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x
    
    
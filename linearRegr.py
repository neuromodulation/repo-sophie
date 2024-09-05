import torch.nn

class EEGRegression(torch.nn.Module): #ab epoch 13500 keine veränderung mehr, loss 0.057739850133657455; epoch 0, loss 0.2543923556804657
    def __init__(self):
        super(EEGRegression, self).__init__()
        self.linear = torch.nn.Linear(6, 1)

    def forward(self, x):
        out = self.linear(x)
        return out
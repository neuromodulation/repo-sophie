import mne
import torch.nn 
import torch.nn.functional as F
import numpy as np
from cnnmodel import *
from linearRegr import *
from read_data import *
import matplotlib.pyplot as plt


model1 = EEGRegression()
lossfunc1 = torch.nn.MSELoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001)
    
model2 = EEGCNNRegr()
lossfunc2 = torch.nn.MSELoss()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
start_epoch = 0
pretrained_model_path = "model_1000_epochs.pth"
try:
    breakpoint = torch.load(pretrained_model_path)
    model2.load_state_dict(breakpoint["model_state_dict"])
    optimizer2.load_state_dict(breakpoint["optimizer_state_dict"])
    start_epoch = breakpoint["epoch"] + 1
    print(f"starting training with epoch {start_epoch}")
except FileNotFoundError:
    print("starting training new")

loss_values = []
q = 1000
for i in range(start_epoch, q): 
    pred_y = model2(x_data)
    loss = lossfunc2(pred_y, y_data)

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    loss_values.append(loss.item())

    if i % 10 == 0:
        print('epoch {}, loss {}'.format(i, loss.item()))

torch.save({"epoch": i,
            "model_state_dict": model2.state_dict(),
            "optimizer_state_dict": optimizer2.state_dict(),
            "loss": loss.item(),}, f"model_{q}_epochs.pth")

#plt.plot(x_data, loss_values) "module" object is not callable
#plt.xlabel("epochs")
#plt.ylabel("loss")
#plt.show()

#laptop too slow to see, if it rlly works, loss reduction of about 0.0011 on 100 epochs (not quite representative)

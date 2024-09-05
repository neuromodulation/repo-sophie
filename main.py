import mne
import torch.nn 
import torch.nn.functional as F
import numpy as np
from cnnmodel import *
from linearRegr import *
from read_data import *


model1 = EEGRegression()
lossfunc1 = torch.nn.MSELoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001)
    
model2 = EEGCNNRegr()
lossfunc2 = torch.nn.MSELoss()
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

for i in range(100): 
    pred_y = model2(x_data)
    loss = lossfunc2(pred_y, y_data)

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    if i % 10 == 0:
        print('epoch {}, loss {}'.format(i, loss.item()))
#laptop too slow to see, if it rlly works, loss reduction of about 0.0011 on 100 epochs (not quite representative)
#target [1, 1, 130001] & input [1, 1] size not the same

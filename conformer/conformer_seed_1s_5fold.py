"""
EEG conformer

Test SEED data 1 second
perform strict 5-fold cross validation
"""

import argparse
import os

import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

from torchmetrics.functional import r2_score
import torchmetrics.functional as TorchMetrics

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

import mne
from matplotlib import pyplot as plt
from sklearn import model_selection

device = torch.device("mps") if torch.has_mps else torch.device("cpu")


cudnn.benchmark = False
cudnn.deterministic = True


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        kernel_size_num_channels = 6

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, 8, (1, 125), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (22, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 8)),
            nn.Dropout2d(0.5),
        )

        self.shallownet = nn.Sequential(
            nn.Conv2d(
                1, emb_size, (1, 25), (1, 1)
            ),  # add emb_size channels, and 25 across the time axis
            nn.Conv2d(emb_size, emb_size, (kernel_size_num_channels, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(
                (1, 75), (1, 15)
            ),  # kernel size (across last two dims), stride
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape

        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(190, 1, 1, 1), nn.LeakyReLU(0.2), nn.Dropout(0.5)
        )
        self.clshead = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
        self.clshead_fc = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Linear(2700, 32),  # not sure how to get to first dimension
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),  # 280
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        # out = self.clshead(x)
        # 1000, 1100
        out = self.fc(x)

        return x, out


# ! Rethink the use of Transformer for EEG signal
class ViT(nn.Sequential):
    def __init__(self, emb_size=100, depth=6, n_classes=1, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes),
        )


class ExGAN:
    def __init__(self):
        super(ExGAN, self).__init__()
        self.batch_size = 1000
        self.n_epochs = 100  # 1000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0001  # 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.alpha = 0.0002
        self.dimension = (190, 50)

        self.start_epoch = 0
        # self.root = "/Data/SEED/seed_syh/data_cv5fold/"
        self.root = "/Users/Timon/Documents/iEEG_deeplearning/root_out"

        self.pretrain = False

        self.log_write = open(
            # "/Code/CT/results/D_base_comp/seed/5-fold/real/log_subject%d_fold%d.txt"
            os.path.join(self.root, "log_subject%d_fold%d.txt"),
            "w",
        )

        self.img_shape = (self.channels, self.img_height, self.img_width)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        # self.criterion_l1 = torch.nn.L1Loss()  # .cuda()
        self.criterion_reg = torch.nn.MSELoss()  # .cuda()

        # self.criterion_reg = r2_score()
        # self.criterion_cls = torch.nn.CrossEntropyLoss()  # .cuda()

        # self.model = ViT().cuda()

        self.model = ViT(
            depth=1,
            emb_size=100,
        )
        # self.model = nn.DataParallel(
        #    self.model, device_ids=[i for i in range(len(gpus))]
        # )
        self.model.to(device)

        # self.model = self.model.cuda()

    def get_source_data(self):
        # self.all_data = np.load(
        #     self.root + "S%d_session1.npy" % self.nSub, allow_pickle=True
        # )
        # self.all_label = np.load(
        #     self.root + "S%d_session1_label.npy" % self.nSub, allow_pickle=True
        # )

        # self.all_data = np.array(self.all_data)

        # (trial, conv channel, electrode channel, time samples)
        self.all_data = np.random.randn(
            100, 1, 62, 200
        )  # 100 samples, 62 channels, 200 time samples
        self.all_label = np.expand_dims(np.random.randint(0, 3, 100), 1)

        PATH_RUNS = [
            r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Data/BIDS_Pittsburgh_Gripforce/rawdata/sub-000/ses-right/ieeg/sub-000_ses-right_task-force_run-0_ieeg.vhdr",
            r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Data/BIDS_Pittsburgh_Gripforce/rawdata/sub-000/ses-right/ieeg/sub-000_ses-right_task-force_run-1_ieeg.vhdr",
            r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Dokumente/Data/BIDS_Pittsburgh_Gripforce/rawdata/sub-000/ses-right/ieeg/sub-000_ses-right_task-force_run-2_ieeg.vhdr",
        ]
        self.all_data = []
        self.all_label = []

        for PATH_RUN in PATH_RUNS:
            raw = mne.io.read_raw_brainvision(PATH_RUN, preload=True)
            picks_ = [
                ch
                for ch in raw.ch_names
                if "ECOG" in ch or "MOV_LEFT_CLEAN" in ch or "MOV_RIGHT_CLEAN" in ch
            ]
            raw.pick(picks=picks_)
            raw.resample(500)  # 250
            raw.filter(
                4, 249, picks=[ch for ch in raw.ch_names if "ECOG" in ch]
            )  # 4, 40
            data = raw.get_data()
            self.all_data.append(data[:-2, :])
            self.all_label.append(data[-1, :] + data[-2, :])
        self.all_label = np.expand_dims(np.concatenate(self.all_label), axis=1)
        self.all_data = np.concatenate(self.all_data, axis=1)

        # convert all_data into epochs of shape 250
        # currently there is no overlap
        epoch_length = 500
        epoch_step_length = 25
        self.all_data = np.array(
            [
                self.all_data[:, i : i + epoch_length]
                for i in range(0, self.all_data.shape[1], epoch_step_length)
                if i + epoch_length < self.all_data.shape[1]
            ]
        )
        self.all_data = np.expand_dims(self.all_data, 1)
        self.all_label = np.expand_dims(
            np.array(
                [
                    np.mean(self.all_label[i : i + epoch_length])
                    for i in range(0, self.all_label.shape[0], epoch_step_length)
                    if i + epoch_length < self.all_label.shape[0]
                ]
            ),
            axis=1,
        )

        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        # simple train test split 80% train, 20% test

        self.train_data, self.test_data, self.train_label, self.test_label = (
            model_selection.train_test_split(
                self.all_data,
                self.all_label,
                test_size=0.3,
                random_state=42,
                shuffle=False,
            )
        )

        # standardize
        target_mean = np.mean(self.train_data)
        target_std = np.std(self.train_data)
        self.train_data = (self.train_data - target_mean) / target_std
        self.test_data = (self.test_data - target_mean) / target_std

        return self.train_data, self.train_label, self.test_data, self.test_label

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train(self):
        train_data, train_label, test_data, test_label = self.get_source_data()

        train_data = torch.tensor(train_data, dtype=torch.float32)
        # img = torch.from_numpy(img)
        # label = torch.from_numpy(label + 1)
        train_label = torch.tensor(train_label, dtype=torch.float32)

        dataset_train = torch.utils.data.TensorDataset(train_data, train_label)
        self.dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=self.batch_size, shuffle=False
        )

        # test_data = torch.from_numpy(test_data)
        # test_label = torch.from_numpy(test_label + 1)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)

        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.dataloader_test = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), maximize=True
        )

        test_data = Variable(test_data.type(self.Tensor)).to(device)
        test_label = Variable(test_label.type(self.Tensor)).to(device)

        train_data = Variable(train_data.type(self.Tensor)).to(device)
        train_label = Variable(train_label.type(self.Tensor)).to(device)

        train_loss = []
        test_loss = []
        for e in range(self.n_epochs):
            self.model.train()
            for i, (batch_traindata, batch_trainlabel) in enumerate(
                self.dataloader_train
            ):  #  iterate through batches
                # img = Variable(img.cuda().type(self.Tensor))
                train_samples = Variable(batch_traindata.type(self.Tensor)).to(device)
                # img = self.active_function(img)
                # label = Variable(label.cuda().type(self.LongTensor))
                batch_trainlabel = Variable(batch_trainlabel.type(self.Tensor)).to(
                    device
                )

                tok, train_pred = self.model(train_samples)

                # loss = self.criterion_reg(train_pred, batch_trainlabel)
                loss = TorchMetrics.concordance_corrcoef(train_pred, batch_trainlabel)
                train_loss.append(loss.detach().cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, test_pred = self.model(test_data)

                # loss_test = self.criterion_reg(test_pred, test_label)
                loss_test = TorchMetrics.concordance_corrcoef(test_pred, test_label)

                print(
                    "Epoch:",
                    e,
                    "  Train loss: %.4f" % loss.detach(),
                    "  Test loss: %.4f" % loss_test.detach(),
                    #    "  Train acc: %.4f" % train_acc,
                    #    "  Test acc: %.4f" % acc,
                )

                test_loss.append(loss_test.detach().cpu().numpy())
                # self.log_write.write(str(e) + "    " + str(acc) + "\n")

            # plt.figure()
            # plt.plot(train_loss)
            # plt.show(block=True)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(train_loss)
        plt.title("Train")
        plt.subplot(2, 1, 2)
        plt.plot(test_loss)
        plt.title("Test")
        plt.show(block=True)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(test_pred.cpu().detach().numpy())
        plt.plot(test_label.cpu().detach().numpy())
        # self.model.eval()
        plt.subplot(2, 1, 2)
        Tok, train_pred = self.model(train_data)
        plt.plot(train_pred.cpu().detach().numpy())
        plt.plot(train_label.cpu().detach().numpy())
        plt.title("Train")
        plt.show(block=True)

        from sklearn import metrics

        print(
            metrics.r2_score(
                test_label.cpu().detach().numpy(), test_pred.cpu().detach().numpy()
            )
        )
        print(
            metrics.r2_score(
                train_label.cpu().detach().numpy(), train_pred.cpu().detach().numpy()
            )
        )
        # return Y_true, Y_pred
        # writer.close()
        print("training complete")


def main():
    best = 0
    aver = 0

    exgan = ExGAN()
    exgan.train()

    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))

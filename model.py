import torch
import torch.nn as nn
from collections import OrderedDict
from torch import optim


class SiAudNet(nn.Module):
    criterion = nn.BCEWithLogitsLoss()

    def __init__(self):
        super(SiAudNet, self).__init__()

        dropout = nn.Dropout(p=0.25)
        pool = nn.MaxPool2d(2, 2)
        padding = 1

        self.conv = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(1, 4, 3, padding=padding)),  # 1 * 32 * 80
                ("relu0", nn.ReLU()),
                ("conv1b", nn.Conv2d(4, 8, 3,
                                     padding=padding)),  # 32 * 32 * 80
                ("relu1", nn.ReLU()),
                ("pool1", pool),
                ("conv2", nn.Conv2d(8, 16, 3,
                                    padding=padding)),  # 32 * 16 * 40
                ("relu2", nn.ReLU()),
                ("pool2", pool),
                ("conv3", nn.Conv2d(16, 32, 3,
                                    padding=padding)),  # 32 * 8 * 20
                ("pool3", pool),
                ("relu3", nn.ReLU()),  # 32 * 4 * 10
            ]))

        self._end_size = 32 * 4 * 10

        self.linear = nn.Sequential(
            OrderedDict([("fc1", nn.Linear(self._end_size, 512)),
                         ("relu1", nn.ReLU())]))

        self.classifier = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(512, 64)),
                ("relu1", nn.ReLU()),
                ("dropout1", dropout),
                ("fc2", nn.Linear(64, 1)),
            ]))

    def forward(self, data: (torch.Tensor, torch.Tensor)):
        (sample_a, sample_b) = data

        sample_a = self.conv(sample_a)
        sample_a = sample_a.view(-1, self.end_size)  # flatten
        sample_a = self.linear(sample_a)

        sample_b = self.conv(sample_b)
        sample_b = sample_b.view(-1, self.end_size)  # flatten
        sample_b = self.linear(sample_b)

        res = torch.abs(sample_b - sample_a)
        res = self.classifier(res)
        res = res.view(-1)
        return res

    def load_dict(self, file_name: str = "model_siaudnet.pt"):
        self.load_state_dict(torch.load(file_name))
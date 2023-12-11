import torch
import torch.nn as nn


class S_Row(nn.Module):
    # 行方向的下采样 结果为H/4*W/32*C
    def __init__(self):
        super(S_Row, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class S_Col(nn.Module):
    # 列方向的下采样 结果为H/32*W/4*C
    def __init__(self):
        super(S_Col, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
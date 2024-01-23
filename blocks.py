import torch.nn as nn
import torch.nn.functional as F

import torch
from conv_lstm import ConvLSTMLayer

# -- For Synthesis -- #


class EncodingBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(EncodingBlock, self).__init__()
        self.conv_lstm = ConvLSTMLayer(in_chans, in_chans, kernel_size)
        self.batch_norm_1 = nn.BatchNorm2d(in_chans)  # could use 3d or batch-wise
        self.conv2d = nn.Conv2d(
            in_chans, out_chans, kernel_size, padding=kernel_size // 2
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_chans)
        self.max_pool = nn.MaxPool2d(2)

        self.out_chans = out_chans

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = self.conv_lstm(x)

        # batch-wise input of spatial time-series
        x = x.reshape(b * t, c, h, w)

        x = F.relu(self.batch_norm_1(x))

        x = self.conv2d(x)

        res = x

        x = self.max_pool(x)

        # reshaping to preserve time-series data
        x = x.reshape(b, t, self.out_chans, h // 2, w // 2)
        return x, res


class DecodingBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(DecodingBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv2d_1 = nn.Conv2d(
            in_chans, in_chans, kernel_size, padding=kernel_size // 2
        )
        self.batch_norm_1 = nn.BatchNorm2d(in_chans)
        self.conv2d_2 = nn.Conv2d(
            in_chans, out_chans, kernel_size, padding=kernel_size // 2
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_chans)

        self.out_chans = out_chans

    def forward(self, x, res):
        bt, c, h, w = x.shape

        x = self.up_sample(x)
        x = self.conv2d_1(x) + res
        x = F.relu(self.batch_norm_1(x))
        x = self.conv2d_2(x)
        x = F.relu(self.batch_norm_2(x))

        return x


# -- For Refinement -- #


class ChannelAttention(nn.Module):
    def __init__(self, in_chans, r=3):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, in_chans // r, 1),
            nn.ReLU(),
            nn.Conv2d(in_chans // r, in_chans, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f_in = x
        x = self.gap(x)
        a_c = self.conv(x)

        return f_in * a_c


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        F_in = x
        avg_out = torch.mean(x, dim=1, keepdim=True)  # channel wise average pool
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # channel wise max pool
        x = torch.cat([avg_out, max_out], dim=1)  # channel-wise concatenation
        x = self.conv(x)
        A_s = F.sigmoid(x)

        return F_in * A_s


class AttentionBlock(nn.Module):
    def __init__(self, num_chans=9, kernel_size=3, padding=1):
        super(AttentionBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding),
        )
        self.ca_block = ChannelAttention(num_chans)
        self.sa_block = SpatialAttention()

    def forward(self, x):
        F_in = x
        x = self.conv_layers(x)
        x = self.ca_block(x)
        x = self.sa_block(x)
        return x + F_in

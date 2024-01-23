import torch.nn as nn

import torch
from blocks import AttentionBlock


class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()
        self.conv_in = nn.Conv2d(9, 9, kernel_size=3, padding=1)
        self.attention_blocks = nn.Sequential(
            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock(),
            AttentionBlock(),
        )
        self.conv_out = nn.Conv2d(9, 3, kernel_size=3, padding=1)

    def forward(self, interpolated_frame, I_2, I_3):
        x = torch.cat([interpolated_frame, I_2, I_3], dim=1)
        x = self.conv_in(x)

        x = x + self.attention_blocks(x)
        x = self.conv_out(x)

        x = x + interpolated_frame

        return x

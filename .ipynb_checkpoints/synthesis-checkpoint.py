import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import EncodingBlock, DecodingBlock

class SynthesisNet(nn.Module):
    def __init__(self):
        
        super(SynthesisNet, self).__init__()
        # Downsampling
        self.encoding_block1 = EncodingBlock(3, 64)
        self.encoding_block2 = EncodingBlock(64, 128)
        self.encoding_block3 = EncodingBlock(128, 256)
        self.encoding_block4 = EncodingBlock(256, 512)
        
        # Middle ConvBlock
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        
        #Upsampling
        self.decoding_block1 = DecodingBlock(512, 256)
        self.decoding_block2 = DecodingBlock(256, 128)
        self.decoding_block3 = DecodingBlock(128, 64)
        self.decoding_block4 = DecodingBlock(64, 3)
        
        # End ConvBlock
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=3, padding=1),
            nn.Conv2d(6, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        
        # getting frames 1 and 2 for linear interpolation
        linear_interpolation = (x[:, 1, :, :, :] + x[:, 2, :, :, :])/2
        
        #encoding with residual connections
        x, res1 = self.encoding_block1(x)
        x, res2 = self.encoding_block2(x)
        x, res3 = self.encoding_block3(x)
        x, res4 = self.encoding_block4(x)
        
        # passing in time-steps one at a time through conv block
        b, time_steps, c, h, w = x.shape
        t_out = torch.zeros(time_steps, b, c, h, w).to(x.device)
        
        for t in range(time_steps):
            x_t = self.conv_block_1(x[:, t, :, :, :])
            t_out[t] = x_t
        
        # decoding with residual connections
        x = t_out.permute(1, 0, 2, 3, 4)
        x = self.decoding_block1(x, res4)
        x = self.decoding_block2(x, res3)
        x = self.decoding_block3(x, res2)
        x = self.decoding_block4(x, res1)
        
        x = torch.cat([x[:, t, :, :, :] for t in range(time_steps)], dim=1)
        x = self.conv_block_2(x)
        
        x = x + linear_interpolation
        return x
                       
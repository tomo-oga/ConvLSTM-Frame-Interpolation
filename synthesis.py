import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import EncodingBlock, DecodingBlock

class SynthesisNet(nn.Module):
    def __init__(self):
        
        super(SynthesisNet, self).__init__()
        # Downsampling
        
        self.encoding_block1 = EncodingBlock(3, 32)
        self.encoding_block2 = EncodingBlock(32, 64)
        self.encoding_block3 = EncodingBlock(64, 128)
        self.encoding_block4 = EncodingBlock(128, 256)
        
        # Middle ConvBlock
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        
        #Upsampling
        self.decoding_block1 = DecodingBlock(256, 128)
        self.decoding_block2 = DecodingBlock(128, 64)
        self.decoding_block3 = DecodingBlock(64, 32)
        self.decoding_block4 = DecodingBlock(32, 3)
        
        # End ConvBlock
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=3, padding=1),
            nn.Conv2d(6, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        
        # getting frames 1 and 2 for linear interpolation
        linear_interpolation = (x[:, 1, :, :, :] + x[:, 2, :, :, :])/2
        x, res1 = self.encoding_block1(x)
        x, res2 = self.encoding_block2(x)
        x, res3 = self.encoding_block3(x)
        x, res4 = self.encoding_block4(x)
        
        # passing in time-steps batch-wise through conv block
        b, t, c, h, w = x.shape
        x = x.reshape(b*t, c, h, w)
        x = self.conv_block_1(x)
            
        x = self.decoding_block1(x, res4)
        x = self.decoding_block2(x, res3)
        x = self.decoding_block3(x, res2)
        x = self.decoding_block4(x, res1)
        
        _, c, h, w = x.shape
        x = x.reshape(b, t, c, h, w)
        x = torch.cat([x[:, i, :, :, :] for i in range(t)], dim=1)
        x = self.conv_block_2(x)
        
        x = x + linear_interpolation
        return x
                       
                       
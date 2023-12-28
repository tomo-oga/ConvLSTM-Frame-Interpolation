import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_lstm import ConvLSTMLayer

# -- For Synthesis -- #

class EncodingBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(EncodingBlock, self).__init__()
        self.conv_lstm = ConvLSTMLayer(in_chans, in_chans, kernel_size)
        self.conv2d = nn.Conv2d(in_chans, out_chans, kernel_size, padding=kernel_size//2)
        self.max_pool = nn.MaxPool2d(2)
        
        self.out_chans = out_chans
    
    def forward(self, x):
        b, seq_len, _, h, w = x.shape
        x = self.conv_lstm(x)
        t_out = torch.zeros(seq_len, b, self.out_chans, h//2, w//2).to(x.device)
        res_out = torch.zeros(seq_len, b, self.out_chans, h, w).to(x.device)
        for t in range(seq_len):
            x_t = self.conv2d(x[:, t, :, :, :])
            res_out[t] = x_t
            t_out[t] = self.max_pool(x_t)
        return t_out.permute(1, 0, 2, 3, 4), res_out.permute(1, 0, 2, 3, 4)
        

class DecodingBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3):
        super(DecodingBlock, self).__init__()        
        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'bicubic')
        self.conv2d_1 = nn.Conv2d(in_chans, in_chans, kernel_size, padding=kernel_size//2)
        self.conv2d_2 = nn.Conv2d(in_chans, out_chans, kernel_size, padding=kernel_size//2)
        
        self.out_chans = out_chans
    
    def forward(self, x, res):
        #ensuring compatible shapes
        xb, xt, xc, xh, xw = x.shape
        xh, xw = xh*2, xw*2
        assert (xb, xt, xc, xh, xw) == res.shape, f"Input and Residual Connection must have same shape, recieved input: {x.shape}, recieved residual: {res.shape}" 
        
                
        b, seq_len, _, h, w = x.shape
        t_out = torch.zeros(seq_len, b, self.out_chans, h*2, w*2).to(x.device)
        for t in range(seq_len):
            x_t = self.up_sample(x[:, t, :, :, :])
            x_t = self.conv2d_1(x_t) + res[:, t, :, :, :]
            t_out[t] = self.conv2d_2(x_t)
        return t_out.permute(1, 0, 2, 3, 4)

# -- For Refinement -- #

class ChannelAttention(nn.Module):
    def __init__(self, in_chans):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(in_chans, in_chans, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_chans, in_chans, 1),
                                  nn.Sigmoid()
                                 )
        
    def forward(self, x):
        f_in = x
        x = self.gap(x)
        a_c = self.conv(x)
        
        return f_in * a_c

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        F_in = x
        avg_out = torch.mean(x, dim=1, keepdim=True) #channel wise average pool
        max_out, _ = torch.max(x, dim=1, keepdim=True) #channel wise max pool
        x = torch.cat([avg_out, max_out], dim=1) # channel-wise concatenation
        x = self.conv(x)
        A_s = F.sigmoid(x)
        
        return F_in * A_s
        
class AttentionBlock(nn.Module):
    def __init__(self, num_chans=9, kernel_size=3, padding=1):
        super(AttentionBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(num_chans, num_chans, kernel_size=kernel_size, padding=padding)
            )
        self.ca_block = ChannelAttention(num_chans)
        self.sa_block = SpatialAttention()
    
    def forward(self, x):
        F_in = x
        x = self.conv_layers(x)
        x = self.ca_block(x)
        x = self.sa_block(x)
        return x + F_in
        
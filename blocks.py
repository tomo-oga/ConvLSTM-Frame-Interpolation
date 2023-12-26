import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_lstm import ConvLSTMLayer

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
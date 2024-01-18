import torch.nn as nn

import torch
from refinement import RefinementNet
from synthesis import SynthesisNet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.synthesis = SynthesisNet()
        self.refinement = RefinementNet()

    def forward(self, x):
        I_2, I_3 = x[:, 1, :, :, :], x[:, 2, :, :, :]
        x = self.synthesis(x)
        rough_frame = x
        refined_frame = self.refinement(x, I_2, I_3)
        return rough_frame, refined_frame

import torch
import torch.nn

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

def loss_fn(synthesis_frame, refinement_frame, ground_truth):
    l1_loss = nn.L1Loss()
    L_1 = l1_loss(synthesis_frame, ground_truth)
    L_2 = l1_loss(refinement_frame, ground_truth)
    L = 0.5 * L_1 + L_2
    return L

def get_optimizer(model):
    return nn.Adam(model.parameters(), lr=1e-4)
    
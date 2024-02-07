import os
import torch
import torch.nn as nn
import time
import gc
from synthesis import SynthesisNet
from refinement import RefinementNet
from datasets import Vimeo90kDatasetTrain, CustomTransform
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

torch.manual_seed(42) #setting seed for reproducibility. NOTE: see cuDNN non-determinism

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.mps.empty_cache()
    # torch.mps.reset_max_memory_allocated()
    torch.mps.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.mps.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))


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

l1_loss = nn.L1Loss()

def loss_fn(rough_frame, refined_frame, ground_truth):
    L_1 = l1_loss(rough_frame, ground_truth)
    L_2 = l1_loss(refined_frame, ground_truth)
    L = 0.5 * L_1 + L_2
    return L

device = torch.device('mps')

dataset = Vimeo90kDatasetTrain(transform = CustomTransform())
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_history = []

if not os.path.exists('model_checkpoints'):
        os.makedirs('model_checkpoints')
    
for epoch in range(0, 150):
    total_loss = 0.0
    num_batches = 0

    start_timer()
    for x_set, y_set in dataloader:
        
        x_set, y_set = x_set.to(device), y_set.to(device)
        x_set, y_set = x_set.permute(0, 1, 4, 2, 3), y_set.permute(0, 3, 1, 2)
        
        optimizer.zero_grad()

        
        rough_frame, refined_frame = model(x_set)
        loss = loss_fn(rough_frame, refined_frame, y_set)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    
    if ((epoch+1) % 5 == 0):
        torch.save(model, f'model_checkpoints/checkpoint-epoch-{epoch+1}')
        
    avg_loss = total_loss/num_batches
    end_timer_and_print(f"Epoch {epoch+1}/150 | loss: {avg_loss:.4f}")
    loss_history.append(avg_loss)
    


torch.save(model, 'model.pth')
np.save('loss_history.npy', loss_history)
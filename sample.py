import torch
import numpy as np

from vrnn import VRNN
from utils import plot
from config import device

model = VRNN(2, 256, 196, 1)
model.to(device)
model.load_state_dict(torch.load("model_weights/32_p1-2_vrnn-backup.pth"))

# hard coded _max and _min values and must be recomputed if sampling from new model
_max = np.array([5.9997, 5.9973])
_min = np.array([5.0002, 5.0005])

for i in range(10):
    traj = model.sample(32)
    traj = (traj * _max) + _min
    data = traj.numpy()
    plot(str(i), data[:,0:2])

print("Done!")
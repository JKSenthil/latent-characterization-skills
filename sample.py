import torch
import numpy as np

from vrnn import VRNN
from plot import plot
from utils import device

model = VRNN(6, 128, 30, 1)
model.to(device)
model.load_state_dict(torch.load("model_weights/32_1_vrnn.pth"))

# hard coded _max and _min values and must be recomputed if sampling from new model
_max = np.array([  5.9993,   5.9926, 106.5546,  25.7080,  25.4740,  63.3103])
_min = np.array([  5.0007,   5.0005, -73.1795, -14.4816, -13.0667, -28.8682])

for i in range(100):
    traj = model.sample(32)
    traj = (traj * _max) + _min
    data = traj.numpy()
    plot("plots/" + str(i), data[:,0:2])

print("Done!")
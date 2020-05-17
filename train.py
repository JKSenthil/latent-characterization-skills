import torch
import numpy as np

from vrnn import VRNN
from torch.utils import data
from config import WEIGHTS_DIR, device
from preprocess import preprocess, normalize

def train():
    print("Starting!")
    model = VRNN(6, 256, 196, 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trajectories = preprocess(None, 32)
    trajectories, _max, _min = normalize(trajectories)
    trajectories = torch.from_numpy(trajectories).float().to(device)

    __max = torch.tensor(_max).float().to(device)
    __min = torch.tensor(_min).float().to(device)

    training_generator = data.DataLoader(trajectories, batch_size=256, shuffle=True)
    validation_generator = data.DataLoader(trajectories, batch_size=256, shuffle=False) # for computing overall loss

    for epoch in range(1, 2001):
        if epoch % 50 == 0:
            torch.save(model.state_dict(), WEIGHTS_DIR + "/{}".format("32_all_vrnn.pth"))
            print("Model saved successfully!")

        for batch in training_generator:
            optimizer.zero_grad()
            kld_loss , mse_loss = model.forward(batch, __max, __min)
            loss = kld_loss + mse_loss
            loss.backward()
            optimizer.step()
        
        kld_loss = 0
        mse_loss = 0
        with torch.no_grad():
            for batch in validation_generator:
                _kld_loss , _mse_loss = model.forward(batch, __max, __min)
                kld_loss += _kld_loss
                mse_loss += _mse_loss
        kld_loss /= len(validation_generator)
        mse_loss /= len(validation_generator)
        print("Epoch {} kld_loss={}, mse_loss={}".format(epoch, kld_loss, mse_loss))
    
    print("Max", __max, "Min", __min)

train()
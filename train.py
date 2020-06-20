import os
import torch
import numpy as np

from utils import train_val_split
from vrnn import VRNN
from torch.utils import data
from config import WEIGHTS_DIR, device
from preprocess import preprocess, normalize

def train():
    print("Starting!")
    model = VRNN(2, 256, 196, 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    p1_trajectories = preprocess(1, 32)[:,:,0:2]
    p2_trajectories = preprocess(2, 32)[:,:,0:2]
    trajectories = np.concatenate((p1_trajectories, p2_trajectories))
    np.random.shuffle(trajectories) # randomize trajectories for train/val 
    trajectories, _max, _min = normalize(trajectories)
    
    train, test = train_val_split(trajectories, val_split=0.15)
    train = torch.from_numpy(train).float().to(device)
    test = torch.from_numpy(test).float().to(device)

    __max = torch.tensor(_max).float().to(device)
    __min = torch.tensor(_min).float().to(device)

    print("Max", __max, "Min", __min)

    training_generator = data.DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    validation_generator = data.DataLoader(test, batch_size=32, shuffle=False, drop_last=True) # for computing overall loss

    val_loss = float("inf")

    for epoch in range(1, 2001):
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

            if kld_loss + mse_loss < val_loss:
                print("Val_loss lowered from {} to {}. Saving model...".format(val_loss, kld_loss + mse_loss))
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "32_p1-2_vrnn.pth"))
                val_loss = kld_loss + mse_loss
    
train()
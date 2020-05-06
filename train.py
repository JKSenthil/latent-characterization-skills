import torch
from torch.utils import data as torch_data
import numpy as np

from vrnn import VRNN
from utils import device
from preprocess import preprocess, normalize

def train():
    model = VRNN(6, 128, 30, 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    data = preprocess(1, 32)
    data, _max, _min = normalize(data)
    data = torch.from_numpy(data).float().to(device)

    __max = torch.tensor(_max).to(device)
    __min = torch.tensor(_min).to(device)

    training_generator = torch_data.DataLoader(data, batch_size=64, shuffle=True)

    for epoch in range(1, 2001):
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "model_weights/32_1_vrnn.pth")
            print("Model saved successfully!")

        for batch in training_generator:
            optimizer.zero_grad()
            kld_loss , mse_loss = model.forward(batch, __max, __min)
            loss = kld_loss + mse_loss
            loss.backward()
            optimizer.step()
        
        kld_loss, mse_loss = model.forward(data, __max, __min)
        print("Epoch {} kld_loss={}, mse_loss={}".format(epoch, kld_loss, mse_loss))

train()
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import FinalBorealis as dg

## Defining the linear model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        return self.linear(x)

def train(model, opt, loss_fn, train_dl, n_epoch=3, save_to='./trainedModel.pth'):
    ##Iterate through a number of epochs
    for epoch in range(n_epoch):
        ##Training with batches of data
        running_loss = 0.0
        num_iters = len(train_dl)
        for i, data in enumerate(train_dl,0):
            inputs, labels = data
            #First step: Generate predictions
            pred = model(inputs)

            #2nd step: Calculate loss
            loss = loss_fn(pred, labels)
            loss.backward()
            opt.step()

            opt.zero_grad()
            running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"Loss at epoch [{epoch}/{n_epoch}], iteration [{i + 1}/{num_iters}]:{running_loss /2000}")
            running_loss = 0.0
    print(f"Training done after {n_epoch} epochs!")

    torch.save(model.state_dict(), save_to)
    print(f"Trained model is saved to {save_to}.")

def main():
    writer = SummaryWriter()

    dataGen = dg.DataGenerator().generateData()
    print(dataGen[:2])
    #inputs (date/hour, type of ingredient)


    #targets (amount needed/consumed per hour)


    

    model = linearRegression(7,4)
    ##Parameters
    list(model.parameters())

    #Define Loss
    loss_fn = F.mse_loss

    #Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

if __name__ == '__main__':
    main()
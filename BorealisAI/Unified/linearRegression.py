import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import FinalBorealis as dg
import datetime
## Defining the linear model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        return self.linear(x)

def train(model, opt, loss_fn, train_ld, n_epoch=3, save_to='./trainedModel.pth'):
    ##Iterate through a number of epochs
    for epoch in range(n_epoch):
        ##Training with batches of data
        running_loss = 0.0
        num_iters = len(train_ld)
        for i, data in enumerate(train_ld,0):
           
            #print(i)
            inputs, labels = data[:,10:], data[:,:9]
            #First step: Generate predictions
            #print(data)
            pred = model(inputs)
            #print(pred)
            #2nd step: Calculate loss
            loss = loss_fn(pred, labels)
            loss.backward()
            opt.step()

            opt.zero_grad()
            running_loss += loss.item()
            if i % 1014 == 1013:
                print(f"Loss at epoch [{epoch}/{n_epoch}], iteration [{i + 1}/{num_iters}]:{running_loss /2000}")
                running_loss = 0.0
    print(f"Training done after {n_epoch} epochs!")

    torch.save(model.state_dict(), save_to)
    print(f"Trained model is saved to {save_to}.")

def dateToInteger(date):
    #print(str(date))
    date = str(date).split('-')
    #print(date)
    newVal = datetime.datetime(int(date[0]), int(date[1]), int(date[2].split(' ')[0]))
    #print(newVal.strftime('%j'))
    return int(newVal.strftime('%j'))

def pdToTensor(dataframe):
    weathers = ['snowStorm', 'snowy', 'rain', 'freezingRain', 'clear']
    newDF = pd.get_dummies(dataframe, prefix='', prefix_sep='')
    newDF['day'] = newDF.apply(lambda row: dateToInteger(row.name), axis=1)
    newDF['Hour'] = newDF.apply(lambda row: int(str(row.name).split(' ')[1].split(':')[0]), axis=1)
    #print(newDF.columns)
    rows, columns = newDF.shape
    ## Create the new tensor set depending on the sizes of the data frame
    newTensor = torch.zeros(rows, columns)
    for i in range(rows):
        for j in range(columns):
            newTensor[i][j] = torch.tensor(newDF.iloc[i,j])

    return newTensor

def main():
    writer = SummaryWriter()

    #dataGen = dg.DataGenerator().generateData()
    
    data = pd.read_json('dataJson.json')
    dataRefined = dg.filteredByHour(data)
    dataTensor = pdToTensor(dataRefined)
    ## Converting the dataframe to a tensor 
    
    #inputs (date/hour, type of ingredient)
    testSize = int(0.1 * len(dataTensor))
    trainSize = len(dataTensor)-testSize

    trainSet, testSet = torch.utils.data.random_split(dataTensor, [trainSize, testSize])

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, num_workers=0)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, num_workers=0)
    
    print(f"Training set consist of {len(trainSet)}, and the test set consists of {len(testSet)}")


    model = linearRegression(6,9)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    #print(model.parameters)
    train(model, opt, loss_fn, trainLoader, n_epoch=500)

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import nni
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import time

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        return self.linear(x)

def train(model, opt, loss_fn, train_ld, n_epoch=3):
    ##Iterate through a number of epochs
    for epoch in range(n_epoch):
        ##Training with batches of data
        running_loss = 0.0
        for i, data in enumerate(train_ld,0):
            inputs, labels = data[:,:16], data[:,16:]
            #First step: Generate predictions
           
            pred = model(inputs)
           
            #2nd step: Calculate loss
            loss = loss_fn(pred, labels)
            loss.backward()
            opt.step()

            opt.zero_grad()
            running_loss += loss.item()
            if i % 500 == 499:
                nni.report_intermediate_result(running_loss)
                running_loss = 0.0

def test(testLoader, model):
    #model = linearRegression(16,1)
    #model.load_state_dict(torch.load(load_pretrained_weights))

    offset = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            #print(data)
            predictions = model(data[:,:16]).numpy()
            offset += mean_squared_error(data[:,16:], predictions)
            total += 1
            
    result = offset / total
    #print(result)
    nni.report_final_result(result) 

def main(args):
    ## Load the data
    data1, data2, data3 = pd.read_json('../Data/data1.json'), pd.read_json('../Data/data2.json'), pd.read_json('../Data/data3.json')
    frames = [data1, data2, data3]
    data = pd.concat(frames, sort=False)
    outputNorm = preprocessing.normalize([np.array(data['output'])])
    data['output'] = outputNorm[0].tolist()

    data = torch.from_numpy(data.values)
    data = data.float()
    #print(data)
    ## Divide the data
    testSize = int(0.1 * len(data))
    trainSize = len(data)-testSize
    
    trainSet, testSet = torch.utils.data.random_split(data, [trainSize, testSize])
    ## Load the data into tensor
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args['batch_size'], num_workers=0)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=args['batch_size'], num_workers=0)

    ## Model and parameters
    model = linearRegression(16,1) ## model = linearRegreggion(nn.Regression)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=args['lr'])

    start = time.time()
    train(model, opt, loss_fn, trainLoader, n_epoch=args['epochs'])
    run_time = time.time() - start
    #print(run_time)
    nni.report_intermediate_result(run_time)
    test(testLoader, model)

if __name__ == '__main__':
    params = {'batch_size':16, 'epochs': 100, 'lr':1e-5}
    params = nni.get_next_parameter()
    main(params)
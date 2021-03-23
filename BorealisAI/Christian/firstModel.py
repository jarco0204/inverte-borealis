import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(( 0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

## training set and data loader
trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=0)

## Test dataset and tada loader to evaluate the performance of the trained model
testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=0)
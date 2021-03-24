import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.ReLU(),
        )
    def forward(self, x):
        features = self.feature_net(x)
        features = features.view(-1,16 * 5 * 5)
        return self.classifier(features)

def imshow(img, label, classes):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(classes[label])
    plt.show()

def train(model, optimizer, criterion, trainLoader, n_epoch=3, save_to='./trained_cifar10.pth'):
    #Loop multiple times 
    for epoch in range(n_epoch):

        running_loss = 0.0
        num_iters = len(trainLoader)
        for i, data in enumerate(trainLoader, 0):
            #data is a list of [inputs, labels]
            inputs, labels = data

            #zero the parameter
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #print stats
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"Loss at epoch [{epoch}/{n_epoch}], iteration [{i + 1}/{num_iters}]: {running_loss / 2000}")
                running_loss = 0.0
    print(f"Training is donde after {n_epoch} epochs!")

    torch.save(model.state_dict(), save_to)
    print(f"Trained model is saved to {save_to}.")

def test(testLoader, load_pretrained_weights='./trained_cifar10.pth'):
    model= Classifier()
    model.load_state_dict(torch.load(load_pretrained_weights))

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testLoader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data,  1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on the test images: {100 * correct / total} %")

def per_class_accuracy(testLoader, classes, batch_size=4 ,load_pretrained_weights='./trained_cifar10.pth'):
    model = Classifier()
    model.load_state_dict(torch.load(load_pretrained_weights))

    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    with torch.no_grad():
        for images, labels in testLoader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            correct = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    for i in range(10):
        print(f"Accuracy of {classes[i]}:{100 * class_correct[i] / class_total[i]} %")
def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(( 0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ## training set and data loader
    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=0)

    ## Test dataset and tada loader to evaluate the performance of the trained model
    testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=0)

    print(f"Training set consists of {len(trainSet)} samples & test set consists of {len(testSet)}.")
    image_0 , label_0 = trainSet[0]
    c, w, h = image_0.shape
    print(f"The first normalized sample of the training data of shape {(c, w, h)} with label {label_0}:\n")
    print(image_0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    imshow(image_0,label_0, classes)

    cifar10_classifier = Classifier()
    #print(cifar10_classifier)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.001, momentum=0.9)
    #print(optimizer)

    #train(model=cifar10_classifier, optimizer=optimizer, criterion=criterion, trainLoader=trainLoader, n_epoch=5)
    test(testLoader=testLoader)
    per_class_accuracy(testLoader=testLoader,classes=classes)

if __name__ == '__main__':
    main()
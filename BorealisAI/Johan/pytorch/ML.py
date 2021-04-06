import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

"""
    PyTorch has created a package called torchvision for vision tasks that has data loaders for datasets such as ImageNet, CIFAR10, MNIST, etc. It also provides tools for transforming the data or writing custom datasets and data loaders.

    CIFAR10 contains data for 10 classes/categories of objects.

    Train on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move model to GPU
    cifar10_classifier = cifar10_classifier.to(device)
    # move first sample to GPU as an example
    image_0, label_0 = image_0.to(device), label_0.to(device)
"""

# global
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def createDataset():
    # use the following to normalize the data and convert to tensors
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # create training dataset and data loader
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=0
    )

    # create test dataset and data loader to evaluate the performance of the trained model
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=0
    )

    # printing some stats about the data
    print(
        f"Training set consists of {len(trainset)} samples & test set consists of {len(testset)}."
    )
    image_0, label_0 = trainset[0]
    c, w, h = image_0.shape
    print(
        f"The first normalized sample of the training data of shape {(c, w, h)} with label {label_0}:\n"
    )
    # print(image_0)
    # imshow(image_0, label_0)
    return [trainset, trainloader, testset, testloader]


"""
Because we have normalized the image, we need to unnormalize it first before plotting
"""


def imshow(img, label):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(classes[label])
    plt.show()


"""
    Class to create the ML model (classifier)
"""


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
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.feature_net(x)
        features = features.view(-1, 16 * 5 * 5)
        return self.classifier(features)


def main():
    data = createDataset()
    cifar10_classifier = Classifier()

    # Loss function
    criterion = nn.CrossEntropyLoss()  # standard choice

    # Optimizer
    optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.001, momentum=0.9)

    train(
        model=cifar10_classifier,
        optimizer=optimizer,
        criterion=criterion,
        trainloader=data[1],
    )
    test(testloader=data[3])

    per_class_accuracy(testloader=data[3])


"""
    There are a lot of tools for keeping track of loss functions
    Tensorboard is one of such tools
    https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
"""


def train(
    model, optimizer, criterion, trainloader, n_epoch=3, save_to="./trained_cifar10.pth"
):
    # loop over the dataset multiple times
    for epoch in range(n_epoch):

        running_loss = 0.0
        num_iters = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    f"Loss at epoch [{epoch}/{n_epoch}], iteration [{i + 1}/{num_iters}]: {running_loss / 2000}"
                )
                running_loss = 0.0

    print(f"Training is done after {n_epoch} epochs!")

    torch.save(model.state_dict(), save_to)
    print(f"Trained model is saved to {save_to}.")


def test(testloader, load_pretrained_weights="./trained_cifar10.pth"):
    model = Classifier()
    model.load_state_dict(torch.load(load_pretrained_weights))

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total} %"
    )


batch_size = 4


def per_class_accuracy(testloader, load_pretrained_weights="./trained_cifar10.pth"):
    model = Classifier()
    model.load_state_dict(torch.load(load_pretrained_weights))

    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]} %")


main()

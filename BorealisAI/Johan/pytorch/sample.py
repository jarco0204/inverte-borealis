# importing libraries
import numpy as np
import torch

"""
    Pytorch was installed with:
    conda install pytorch torchvision torchaudio -c pytorch

    Tensors === numpy.arrays
    Former can be used for GPU

    Other resources:
    • https://github.com/jcjohnson/pytorch-examples
    • https://github.com/yunjey/pytorch-tutorial

"""


def basicInit():
    # initializing a numpy array
    a = np.array(1)
    # initializing a tensor
    b = torch.tensor(1)
    print(a)
    print(b)
    print(type(a))
    print(type(b))


def basicOperations():
    # initializing two tensors
    a = torch.tensor(2)
    b = torch.tensor(1)
    print(a, b)
    # addition
    print(a + b)
    # subtraction
    print(b - a)
    # multiplication
    print(a * b)
    # division
    print(a // b)


def matrixTensor():
    # matrix of zeros
    a = torch.zeros((3, 3))
    print(a)
    print(a.shape)

    # matrix of random numbers
    a = torch.randn(3, 3)


def matrixOperations():
    # initializing two tensors
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    # matrix addition
    print(torch.add(a, b), "\n")
    # matrix subtraction
    print(torch.sub(a, b), "\n")
    # matrix multiplication
    print(torch.mm(a, b), "\n")
    # matrix division
    print(torch.div(a, b))

    # Matrix transpose is one technique which is also very useful while creating a neural network from scratch
    print(torch.t(a))

    # Concatenation
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print(a, "\n")
    print(b)

    # concatenating vertically
    torch.cat((a, b))

    # concatenating horizontally
    torch.cat((a, b), dim=1)

    # Reshaping
    a = torch.randn(2, 4)
    print(a)
    a.shape

    # reshaping tensor
    b = a.reshape(1, 8)
    print(b)
    b.shape

    # convert NP array to tensor
    a = np.array([[1, 2], [3, 4]])
    print(a, "\n")

    # converting the numpy array to tensor
    tensor = torch.from_numpy(a)
    print(tensor)


def autograd():
    # PyTorch uses a technique called automatic differentiation
    # Records all the operations that we are performing and replays it backward to compute gradients.

    # initializing a tensor
    # Specifying requires_grad as True will make sure that the gradients are stored for this particular tensor whenever we perform some operation on it.
    a = torch.ones((2, 2), requires_grad=True)

    # performing operations on the tensor
    b = a + 5
    c = b.mean()
    print(b, c)

    # Now, the derivative of c with respect to a will be ¼ and hence the gradient matrix will be 0.25

    # back propagating
    c.backward()

    # computing gradients
    print(a.grad)


autograd()


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import dataloader, random_split
import matplotlib.pyplot as plt
import numpy as np

'''
torch.manual_seed(19)
print(
torch.cuda.is_available()
)
'''

# import data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1037),(0.3081))]) # normalize with MNIST mean and standard dev values found online

# gets training and testing data
train_data = datasets.MNIST(root= 'data', train = True, download = True, transform=transform)
test_data = datasets.MNIST(root= 'data', train = False, download = True, transform=transform)

# print(train_data)

#CNN

#MLP
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

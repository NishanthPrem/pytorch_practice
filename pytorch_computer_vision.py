#%% Pytorch CV

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Loading the datasets

train_data = datasets.MNIST(root="data",
                            train=True,
                            download=True,
                            transform=ToTensor(),
                            target_transform=None)

test_data = datasets.MNIST(root="data",
                            train=False,
                            download=True,
                            transform=ToTensor())

#%% Visualize 5 labels

for i in range(len(train_data.classes)):
    img = train_data[i][0]
    print(img.shape)
    img_squeeze = img.squeeze()
    print(img_squeeze.shape)
    label = train_data[i][1]
    plt.figure(figsize=(3, 3))
    plt.imshow(img_squeeze, cmap="gray")
    plt.title(label)
    plt.axis(False)
  
#%% Loading the data into DataLoader

BATCH_SIZE = 32

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

#%% Building the model

class MNISTModel(nn.Module):
    def __init__(self, input_shape, output_features, hidden_layers):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_shape, hidden_layers, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(hidden_layers, hidden_layers,3, padding=1)
        self.conv4 = nn.Conv2d(hidden_layers, hidden_layers,3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_layers*7*7, output_features)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.flatten(x)
        
        return self.fc1(x)
torch.manual_seed(42)
mnist_model = MNISTModel(1, len(train_data.classes), 10).to(device)

#%%
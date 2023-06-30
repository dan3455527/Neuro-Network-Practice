import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 16, 5, 1, 2),
      nn.ReLU(),
      nn.MaxPool2d(2)
    ) 
    self.conv2 = nn.Sequential(
      nn.Conv2d(16, 32, 5, 1, 2),
      nn.ReLU(),
      nn.MaxPool2d(2)
    ) 
    self.fc1 = nn.Linear(32*7*7, 10)
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    return x 
  
class FCN(nn.Module):
  def __init__(self):
    super(FCN, self).__init__()
    self.layer1 = nn.Linear(28*28, 100)
    self.layer2 = nn.Linear(100, 50)
    self.layer3 = nn.Linear(50, 10)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = x.view(-1, 28*28)
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    x = self.relu(x)
    x = self.layer3(x)
    return x
    
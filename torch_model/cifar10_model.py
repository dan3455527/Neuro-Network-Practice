import torch
import torch.nn as nn
import torch.nn.functional as F
  

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.convblock1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2, 2)
    ) 
    self.convblock2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2, 2)
    )
    self.convblock3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2)
    )
    self.fc1 = nn.Linear(128*8*8, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 10)
    pass
  
  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

class CNN2(nn.Module):
  def __init__(self):
    super(CNN2, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
      nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.Dropout(0.25),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
      nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(64),
      nn.Dropout(0.25),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
    )
    self.flatten = nn.Flatten()
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(0.25)
    self.fc1 = nn.Linear(64*8*8, 1600)
    self.fc2 = nn.Linear(1600, 512)
    self.fc3 = nn.Linear(512, 128)
    self.fc4 = nn.Linear(128, 32)
    self.fc5 = nn.Linear(32, 10)
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.drop(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.drop(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.drop(x)
    x = self.relu(x)
    x = self.fc4(x)
    x = self.drop(x)
    x = self.relu(x)
    x = self.fc5(x)
    return x
    
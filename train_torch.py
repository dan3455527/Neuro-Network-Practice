import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_model.model import CNN, FCN
from torchsummary import summary
import matplotlib.pyplot as plt
from keras.datasets import mnist
from utils.preprocessing import get_split_sampler
from utils.trainer import TrainCompile, TestCompile
from utils.visulizaition import show_train_history, show_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

class Trainset(Dataset):
  def __init__(self, train_data: np.ndarray, train_target: np.ndarray) -> None:
    train_data = np.expand_dims(train_data, axis=1)
    train_data = train_data / 255

    self.train_data = torch.from_numpy(train_data)
    self.train_target = torch.from_numpy(train_target)
    self.n_samples = train_data.shape[0] 
    pass
    
  def __getitem__(self, index):
    return self.train_data[index], self.train_target[index]

  def __len__(self):
    return self.n_samples

dataset = Trainset(X_train, Y_train)
test_set = Trainset(X_test, Y_test)
# data split
train_sampler, val_sampler = get_split_sampler(dataset, 0.2)
batch_size = 1000

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

model = CNN().to(device)
summary(model, dataset.train_data.shape[1:], batch_size=-1)
model.train()


# train
n_epoch=50
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

compiler = TrainCompile(
  model=model,
  train_dataloader=train_loader,
  loss_func=loss_func,
  optimizer=optimizer,
  n_epochs=n_epoch,
  device=device,
  val_dataloader=val_loader,
  verbose=0
)

test_compiler = TestCompile(model, X_test, Y_test, device)

history = compiler.fit()

pred, acc = test_compiler.evaluate()
conf_mat = test_compiler.get_confusion_matrix()
show_train_history(history)
show_confusion_matrix(conf_mat)

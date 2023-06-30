#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from keras.datasets import cifar10
from utils.preprocessing import get_split_sampler
from torch_model.cifar10_model import *
from utils.trainer import TestCompile, TrainCompile
from utils.visulizaition import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

class Trainset(Dataset):
  def __init__(self, train_data, train_target):
    train_data = np.reshape(train_data, (train_data.shape[0], 3, 32, 32))
    train_data = train_data / 255
    train_target = train_target.squeeze()
    
    self.train_data = torch.from_numpy(train_data)
    self.train_target = torch.from_numpy(train_target)
    self.n_samples = len(self.train_data)
    pass

  def __getitem__(self, index):
    return self.train_data[index], self.train_target[index]

  def __len__(self):
    return self.n_samples
  
dataset = Trainset(X_train, Y_train)
train_sampler, val_sampler = get_split_sampler(dataset, 0.2)
batch_size = 256

trainloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_sampler)

model = CNN2()
model = model.to(device)
n_epoch = 30
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
summary(model, (3, 32, 32), batch_size=batch_size, device="cuda")
#%%
compiler = TrainCompile(
  model=model,
  train_dataloader=trainloader,
  loss_func=loss_func,
  optimizer=optimizer,
  n_epochs=n_epoch,
  device=device,
  val_dataloader=val_loader,
  verbose=0
)
history = compiler.fit(is_checkpoint=False, lr_decay=False)
show_train_history(history)
#%%
test_compiler = TestCompile(model, X_test, Y_test, device='cpu')
pred, acc = test_compiler.evaluate()
conf_mat =test_compiler.get_confusion_matrix()
show_confusion_matrix(conf_mat)
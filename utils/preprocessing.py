import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class TrainSet(Dataset):
  def __init__(self, train_data, train_target):
    pass
  
  def __getitem__(self, index):
    pass
  
  def __len__(self):
    pass


def get_split_sampler(datasets, split_ratio, shuffle=True, random_seed=1234):
  dataset_size = len(datasets)
  indices = list(range(dataset_size))
  split = int(np.floor(dataset_size * split_ratio))
  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_indices)
  val_sampler = SubsetRandomSampler(val_indices)
  return train_sampler, val_sampler
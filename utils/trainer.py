import torch
import torch.nn.functional as F
import numpy as np
import time

class TrainCompile():
  def __init__(self, model, train_dataloader, loss_func, optimizer, n_epochs, device=None, val_dataloader=None, verbose=0):
    self.model = model
    self.train_dataloader = train_dataloader
    self.loss_func = loss_func
    self.optimizer = optimizer
    self.n_epochs = n_epochs
    if device == None:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = device
    self.val_dataloader = val_dataloader
    self.verbose = verbose
    pass
  
  def train_iteration_loop(self):
    self.model.train()
    iter_loss_ls = []
    iter_acc_ls = []
    for i , (feature, target) in enumerate(self.train_dataloader):
      feature = feature.to(self.device, dtype=torch.float)
      target = target.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.model(feature)
      iter_loss = self.loss_func(outputs, target)
      iter_loss.backward()
      self.optimizer.step()
      pred = torch.argmax(F.softmax(outputs), dim=1)
      num_acc = sum(pred==target).item()
      iter_acc = num_acc / len(target)
      iter_loss_ls.append(iter_loss.item())
      iter_acc_ls.append(iter_acc)
      # display verbose
      if self.verbose==1:
        print(f"step {i+1}/{len(self.train_dataloader)}, loss : {iter_loss:.4f}, acc : {iter_acc:.4f}", end="\r", flush=True)
        pass
      pass

    avg_loss = np.mean(np.array(iter_loss_ls))
    avg_acc = np.mean(np.array(iter_acc_ls))
    return avg_loss, avg_acc
    
  def validation_iteration_loop(self):
    self.model.eval()
    iter_val_loss_ls = []
    iter_val_acc_ls = []
    for i, (feature, target) in enumerate(self.val_dataloader):
      feature = feature.to(self.device, dtype=torch.float)
      target = target.to(self.device)
      outputs = self.model(feature)
      iter_val_loss = self.loss_func(outputs, target)
      pred = torch.argmax(F.softmax(outputs), dim=1)
      num_acc = sum(pred==target).item()
      iter_val_acc = num_acc / len(target)
      iter_val_loss_ls.append(iter_val_loss.item())
      iter_val_acc_ls.append(iter_val_acc)
    avg_val_iter_loss = np.mean(np.array(iter_val_loss_ls))
    avg_val_iter_acc = np.mean(np.array(iter_val_acc_ls))
    return avg_val_iter_loss, avg_val_iter_acc

  def verbose_display(self, epoch, loss, val_loss, acc, val_acc):
    loss, val_loss, acc, val_acc = \
      np.mean(loss), np.mean(val_loss), np.mean(acc), np.mean(val_acc)
    display_text = f"""Epoch {epoch+1}/{self.n_epochs} \nloss: {loss:.4f}, val_loss: {val_loss:.4f}, acc: {acc:.4f}, val_acc: {val_acc:.4f}\n{"-"*70}"""
    return display_text

  def fit(self):
    history = {
      "loss" : np.array([]),
      "val_loss" : np.array([]),
      "acc" : np.array([]),
      "val_acc" : np.array([])
    } 
    for epoch in range(self.n_epochs):
      loss, acc = self.train_iteration_loop()
      if self.val_dataloader != None:
        val_loss, val_acc = self.validation_iteration_loop()
      history["loss"] = np.hstack((history["loss"], loss))
      history["acc"] = np.hstack((history["acc"], acc))
      history["val_loss"] = np.hstack((history["val_loss"], val_loss))
      history["val_acc"] = np.hstack((history["val_acc"], val_acc))
      display_text = self.verbose_display(epoch, loss, val_loss, acc, val_acc)
      print(display_text)
      pass
    return history

  def get_runtime(self):
    time_sec = time.time()
    
    pass
import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import confusion_matrix

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

  def verbose_display(self, epoch, loss, val_loss, acc, val_acc, runtime):
    loss, val_loss, acc, val_acc = \
      np.mean(loss), np.mean(val_loss), np.mean(acc), np.mean(val_acc)
    display_text = f"""Epoch {epoch+1}/{self.n_epochs} \nloss: {loss:.4f}, val_loss: {val_loss:.4f}, acc: {acc:.4f}, val_acc: {val_acc:.4f}, {runtime}\n{"-"*70}"""
    return display_text

  def fit(self):
    history = {
      "loss" : np.array([]),
      "val_loss" : np.array([]),
      "acc" : np.array([]),
      "val_acc" : np.array([])
    } 
    total_str_time = time.time()
    for epoch in range(self.n_epochs):
      epoch_str_time = time.time()
      loss, acc = self.train_iteration_loop()
      if self.val_dataloader != None:
        val_loss, val_acc = self.validation_iteration_loop()
      history["loss"] = np.hstack((history["loss"], loss))
      history["acc"] = np.hstack((history["acc"], acc))
      history["val_loss"] = np.hstack((history["val_loss"], val_loss))
      history["val_acc"] = np.hstack((history["val_acc"], val_acc))
      epoch_end_time = time.time()
      runtime = self.get_runtime(epoch_str_time, epoch_end_time)
      display_text = self.verbose_display(epoch, loss, val_loss, acc, val_acc, runtime)
      print(display_text)
      pass
    total_end_time = time.time()
    print(f"total train time : {self.get_runtime(total_str_time, total_end_time)}")
    return history

  def get_runtime(self, str_time, end_time):
    inference_time = end_time-str_time
    hours = int(inference_time // 3600)
    minute = int((inference_time % 3600) // 60)
    seconds = int(inference_time % 60)
    inference_time_str = f"{hours:02d}:{minute:02d}:{seconds:02d}"
    return inference_time_str
  


class TestCompile():
  def __init__(self,model, test_data, test_target, device):
    self.model=model
    self.test_data = test_data
    self.norm_test_data_torch = torch.from_numpy(np.expand_dims(test_data, axis=1))/255
    self.test_target = test_target
    self.test_target_torch = torch.from_numpy(test_target)
    self.device=device
    pass
  
  def evaluate(self):
    self.model.eval()
    self.model = self.model.to(self.device)
    outputs = self.model(self.norm_test_data_torch.to(self.device, dtype=torch.float))
    self.pred = torch.argmax(F.softmax(outputs), dim=1)
    num_acc = sum(self.pred==self.test_target_torch.to(self.device))
    self.accuracy = num_acc / len(self.norm_test_data_torch)
    print(f"""Evaluation : \nAccuracy : {self.accuracy:.4f}\n{"-"*70}""")
    return self.pred.detach().cpu().numpy(), self.accuracy.detach().cpu().numpy()
  
  def get_confusion_matrix(self):
    conf_mat = confusion_matrix(self.test_target, self.pred.detach().cpu().numpy()) 
    return conf_mat
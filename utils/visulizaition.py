import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def show_train_history(history):
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))
  ax[0].plot(history["loss"])
  if len(history["val_loss"]) != 0:
    ax[0].plot(history["val_loss"])
  ax[0].set_title("training loss")
  ax[0].set_xlabel("Epoch")
  ax[0].set_ylabel("Loss")
  ax[0].legend(["loss", "val_loss"])

  ax[1].plot(history["acc"])
  if len(history["val_acc"]) != 0:
    ax[1].plot(history["val_acc"])
  ax[1].set_title("training accuracy")
  ax[1].set_xlabel("Epoch")
  ax[1].set_ylabel("Accuracy")
  ax[1].legend(["acc", "val_acc"])
  pass

def show_confusion_matrix(conf_mat):
  fig = plt.figure(figsize=(15, 10))
  df_conf_mat = pd.DataFrame(conf_mat)
  sn.heatmap(
    conf_mat,
    annot=True,
    cmap="Blues",
    cbar=False,
    fmt="g"
  )
  pass

class DisplayPlot():
  def __init__():
    pass
  
  def show_train_history(self):
    pass

  def show_roc_curve(self):
    pass

class DisplayMatrix():
  def __init__():
    pass

  def show_confusion_matrix(self):
    pass
import torch
import numpy as np
import torchvision
!pip install yacs
from yacs.config import CfgNode as CN
import torchvision
import torch.nn as nn
import os
import pandas as pd
import shutil
#創分類資料夾
path = "/kaggle/working/abc_train"
os.mkdir(path)
path = "/kaggle/working/abc_dev"
os.mkdir(path)
path = "/kaggle/working/abc_train/A"
os.mkdir(path)
path = "/kaggle/working/abc_train/B"
os.mkdir(path)
path = "/kaggle/working/abc_train/C"
os.mkdir(path)
path = "/kaggle/working/abc_dev/A"
os.mkdir(path)
path = "/kaggle/working/abc_dev/B"
os.mkdir(path)
path = "/kaggle/working/abc_dev/C"
os.mkdir(path)
#把圖片放入各資料夾
df = pd.read_csv('../input/train-mango/train.csv',encoding = "ISO-8859-1")  
for i in range(df.shape[0]):
    old='../input/train-mango/C1-P1_Train/'+df["image_id"][i]
    new='/kaggle/working/abc_train/'+df["label"][i]
    shutil.copy(old,new)
df = pd.read_csv('../input/train-mango/dev.csv',encoding = "ISO-8859-1")  
for i in range(df.shape[0]):
    old='../input/train-mango/C1-P1_Dev/'+df["image_id"][i]
    new='/kaggle/working/abc_dev/'+df["label"][i]
    shutil.copy(old,new)
!cp -r ../input/hw2mango7/hw2mangos/* ./
import model
import datasets
import config
import figures
!pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!python ../input/hw2mango7/hw2mangos/train.py
!python ../input/hw2mango7/hw2mangos/test.py
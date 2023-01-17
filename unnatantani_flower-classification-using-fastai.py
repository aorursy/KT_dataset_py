%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision.all import *
from fastai.imports import *
from fastai.vision.data import *
from fastai import *
import numpy as np
import fastai
import matplotlib.pyplot as plt
path = Path("/kaggle/input/flowers-recognition/flowers")
path.ls()

np.random.seed(42)
data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, item_tfms=RandomResizedCrop(512, min_scale=0.75),
                                    bs=32,batch_tfms=[*aug_transforms(size=256, max_warp=0), Normalize.from_stats(*imagenet_stats)],num_workers=0)

data.show_batch(nrows=3, figsize=(7,8))
learn = cnn_learner(data, resnet50, metrics=error_rate)
learn.fit_one_cycle(4)
learn.unfreeze()
learn.fit_one_cycle(2, lr_max=slice(1e-5,1e-4)) #Using graph for learn.lr_find()
learn.model = learn.model.cpu()
learn.export("/kaggle/working/export.pkl")
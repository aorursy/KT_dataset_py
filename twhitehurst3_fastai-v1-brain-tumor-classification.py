import numpy as np 

import pandas as pd 

import os,gc,pathlib

from sklearn.metrics import confusion_matrix

from fastai import *

from fastai.vision import *

from fastai.vision.models import *

print(os.listdir("../input"))

import torchvision.models as models
DATA_DIR='../input/brain_tumor_dataset'
os.listdir(f'{DATA_DIR}')
data = ImageDataBunch.from_folder(DATA_DIR, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=8, 

                                  num_workers=0).normalize(imagenet_stats)

print(f'Classes: \n {data.classes}')
data.show_batch(rows=10, figsize=(10,5))
learn = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10,3e-3)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=slice(1e-6,1e-5))
learn.save('stage-2')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
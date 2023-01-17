import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import time

from fastai.vision import *

from fastai.metrics import accuracy
dir_path = Path("../input/intel-image-classification/")
np.random.seed(int(time.time()))
BATCH_SIZE = 64

SIZE = 150
data = ImageDataBunch.from_folder(path=dir_path, 

                                  train='seg_train',

                                  valid='seg_test',

                                  test='seg_pred',

                                  ds_tfms=get_transforms(), 

                                  size=SIZE,

                                  bs=BATCH_SIZE).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.model_dir = "/kaggle/working/"
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1e-4,1e-3))
learn.save('stage-1-res50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1e-5, 1e-4))
learn.save('stage-2-res50')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
interp.plot_top_losses(9, figsize=(10,9))
interp.most_confused(min_val=2)
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *



import os

print(os.listdir("../input/pollendataset/PollenDataset/"))

pd.read_csv("../input/pollendataset/PollenDataset/pollen_data.csv").head()
path=Path("../input/pollendataset/PollenDataset/")

data = ImageDataBunch.from_csv(path, folder='images/',csv_labels='pollen_data.csv',fn_col=1, label_col=2,

                                  ds_tfms=get_transforms(flip_vert=True,do_flip=True, max_warp=0,  max_rotate=0, p_affine=0.5, max_zoom=1.1),

                                  bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')

data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10,1e-2)
learn.save('stage-1')
learn.lr_find()

learn.recorder.plot()
learn.load('stage-1')

learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(5e-5,5e-3 ))
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
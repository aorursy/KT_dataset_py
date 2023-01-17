%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *

base_dir = '../input/cell_images/cell_images/'

base_path = Path(base_dir)

base_path
data = ImageDataBunch.from_folder(base_path,valid_pct=0.1,

                                 train='.',

                                 ds_tfms=get_transforms(max_warp=0,flip_vert=True),

                                 size=128,bs=32,

                                 num_workers=0).normalize(imagenet_stats)



print(f'Classes to classify: \n {data.classes}')

data.show_batch(rows=5,figsize=(7,7))
learner = create_cnn(data,models.resnet50,metrics=accuracy,model_dir='/tmp/model/')

learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(10,max_lr=slice(1e-4,1e-3))

learner.save('stage-1')
learner.recorder.plot_losses()
inter = ClassificationInterpretation.from_learner(learner)

inter.plot_top_losses(9,figsize=(20,20))
inter.plot_confusion_matrix(figsize=(10,10),dpi=75)
learner.unfreeze()

learner.fit_one_cycle(2)
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(5, max_lr=slice(1e-6,1e-3))
learner.recorder.plot_losses()
inter = ClassificationInterpretation.from_learner(learner)

inter.plot_top_losses(9,figsize=(20,20))
inter.plot_confusion_matrix(figsize=(10,10),dpi=75)

learner.save('malaria-fastai-V1')
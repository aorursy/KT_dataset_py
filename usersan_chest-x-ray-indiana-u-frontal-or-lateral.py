%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install "torch==1.4" "torchvision==0.5.0"
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.ndimage



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *
df_all = pd.read_csv('../input/chest-xrays-indiana-university/indiana_projections.csv')

df_all.info()

df_all.head()
df = df_all[['filename', 'projection']]

df.head()
path_img = Path('../input/chest-xrays-indiana-university/images/images_normalized')

path = Path('../working')

#path_img.ls()
learn = None

gc.collect()
bs = 256

tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_df(path_img, df, ds_tfms=tfms, size=256, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
np.random.seed(101)

learn = cnn_learner(data, models.resnet18, metrics=accuracy)
#learn.lr_find()

#learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=3e-3)
#learn.unfreeze()
#learn.fit_one_cycle(3, max_lr=slice(3e-5,3e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
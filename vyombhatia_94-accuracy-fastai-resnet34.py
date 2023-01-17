import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import accuracy
path = "../input/nepali-barnamala-hand-written-data-set/NHWDataset/test"

traindata = ImageDataBunch.from_folder(path, ds_tfms = get_transforms(do_flip=False, flip_vert=False),

                                      valid_pct=0.3, size=256,

                                      bs=16)
traindata.show_batch(row=3)
print(traindata.classes)
learn = cnn_learner(traindata, models.resnet34, metrics = accuracy,  model_dir="/tmp/model/")
learn.model
learn.fit_one_cycle(5)
learn.lr_find()
learn.recorder.plot()
image = open_image("../input/nepali-barnamala-hand-written-data-set/NHWDataset/test/character_14_dhaa/017_02.jpg")
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

image.show()

pred_class, pred_idx, pred_outputs = learn.predict(image)
print(pred_class)
image2 = open_image("../input/nepali-barnamala-hand-written-data-set/NHWDataset/test/character_21_pa/017_07.jpg")
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

image2.show()

pred_class, pred_idx, pred_outputs = learn.predict(image2)

print(pred_class)
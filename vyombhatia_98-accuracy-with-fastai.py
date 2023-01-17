%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fastai.vision import *

from fastai.metrics import accuracy
data = ImageDataBunch.from_folder("../input/cat-and-dog/training_set/",

                                 ds_tfms = get_transforms(do_flip=False, flip_vert=False),

                                 valid_pct=0.2,

                                 size=224,

                                 bs=16)
data.show_batch(row=3)
print(data.classes)
learn = cnn_learner(data, models.resnet34, metrics = accuracy)
learn.model
learn.fit_one_cycle(2)
learn.recorder.plot()
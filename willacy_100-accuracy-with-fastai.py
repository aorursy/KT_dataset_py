import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
path = Path('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images')

path.ls()
path.ls()
data = ImageDataBunch.from_folder(path, train='TRAIN', test='TEST', valid_pct=0.20,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(2)
learn.unfreeze()
learn.model_dir=Path('/kaggle/working')

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(5e-5,5e-4))
learn.save('model')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
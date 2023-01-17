%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *

from fastai.metrics import error_rate

import os
bs = 64
path = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')

path.ls()
np.random.seed(5)

data = ImageDataBunch.from_folder(path, valid = 'test', size=299, bs=bs, ds_tfms=get_transforms()).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(6,6))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, model_dir = '/tmp/model/', metrics=error_rate)
learn.fit_one_cycle(8)
learn.save('step-1-50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(7e-6, 3e-4))
learn.save('step-2-50')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
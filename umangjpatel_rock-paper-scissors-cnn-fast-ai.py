import os

print(os.listdir('../input'))
from fastai import *

from fastai.vision import *
print(os.listdir('../input/rps-cv-images/'))
path = Path('../input/rps-cv-images')

print(path)

print(path.ls())
data = ImageDataBunch.from_folder(

    path=path,

    train=".",

    valid_pct=0.1,

    size=224,

    ds_tfms=get_transforms()

)

data.normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learner = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate], model_dir='/tmp/models/')
learner.lr_find()

learner.recorder.plot()
lr = 1e-03

learner.fit_one_cycle(4, max_lr=slice(lr))
learner.save('stage-1-frozen-resnet34')
learner.recorder.plot_losses()
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(4, max_lr=slice(1e-04))
learner.save('stage-2-unfrozen-resnet34')
learner.recorder.plot_losses()
learner.recorder.plot_lr()
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
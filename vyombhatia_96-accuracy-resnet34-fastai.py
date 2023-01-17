import pandas as pd
from fastai.vision import *
from fastai.metrics import accuracy
path = "../input/dog-data/ours_dog"

data = ImageDataBunch.from_folder(path, ds_tfms = get_transforms(), bs=16, size=256)
data.show_batch()
learn = cnn_learner(data ,models.resnet34 ,model_dir = "/temp/model/", metrics = accuracy)
learn.model
learn.fit_one_cycle(7)
learn.save('stage-1')
learn.load('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, slice(1e-5, 1e-4))
interpreter = ClassificationInterpretation.from_learner(learn)

interpret(ds_type=data)
interpreter.plot_top_losses(4, figsize=(12,12))
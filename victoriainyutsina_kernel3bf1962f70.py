# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from fastai.vision import *
from fastai.metrics import error_rate
import os
np.random.seed(12)

path = Path('/kaggle/input/ships-in-satellite-imagery/shipsnet/shipsnet')
fnames = get_image_files(path)
pat = r'^\D*(\d+)'
tfms = get_transforms(flip_vert=True, # корабль может появиться с любой стороны
                      max_warp=0)     # корабли не должны никак изменяться 

data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=tfms, size=256, bs=64).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")
data.show_batch(rows=3, figsize=(7,6))
lr_find(learn) # ищу самый подходящий лернинг рейт
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-1)) # обучаю с использованием оптимального lr
learn.save('fit')
val_probas, val_labels = learn.get_preds(DatasetType.Valid) # сохраняю предсказание (вероятности и классы)
resnet50_loss = log_loss(data.valid_ds.y.items, val_probas.numpy())
resnet50_loss # бейзлайн пройден
interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(7, 6)) # на чем чаще всего модель ошибается
interp.plot_confusion_matrix()
learn.show_results(rows=3, figsize=(10,10)) # результаты

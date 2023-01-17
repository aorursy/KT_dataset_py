# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



from fastai.vision import *
bs = 64



np.random.seed(2)



data = ImageDataBunch.from_folder(

    '../input/chest_xray/chest_xray', 

    #valid='val',

    valid_pct=0.02,

    size=244,

    ds_tfms=get_transforms(), 

    bs=bs

).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")

learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-3))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(24,24), dpi=60)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *
path = Path("/kaggle/input/aptos2019-blindness-detection")

train_path = path/'train_images'

test_path = path/'test_images'

working_path = Path("/kaggle/working")

output_path = Path("/kaggle/output")
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

model_path = '/tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth'

!cp ../input/resnet152/resnet152-b121ed2d.pth {model_path}
kappa = KappaScore()

kappa.weights = "quadratic"
data = (ImageList.from_csv(path, csv_name='train.csv', folder='train_images', suffix='.png')

                .split_by_rand_pct(valid_pct=0.2, seed=42)

                .label_from_df()

                .transform(get_transforms(flip_vert=True, max_rotate=360.0, max_warp=0.1), size=224)

                .databunch(bs=16, num_workers=os.cpu_count())

                .normalize())
data.show_batch(rows=3, figsize=(7,8))
learn = cnn_learner(data, models.resnet152, metrics=[error_rate, kappa])
learn.model_dir = '/kaggle/working'
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3)
learn.save('stage-1-224')
learn.load('stage-1-224')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-4))
learn.save('stage-2-224')
data = (ImageList.from_csv(path, csv_name='train.csv', folder='train_images', suffix='.png')

                .split_by_rand_pct(valid_pct=0.2, seed=42)

                .label_from_df()

                .transform(get_transforms(flip_vert=True, max_rotate=360.0, max_warp=0.1), size=448)

                .databunch(bs=16, num_workers=os.cpu_count())

                .normalize())
learn = cnn_learner(data, models.resnet152, metrics=[error_rate, kappa])
learn.model_dir = '/kaggle/working'
learn.load('stage-2-224')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, 1e-3)
learn.save('stage-1-448')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(3, max_lr=slice(4e-06, 4e-4))
learn.save('stage-2-448')
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df, path, folder='test_images', suffix='.png'))
import numpy as np

import pandas as pd

import os

import scipy as sp

from functools import partial

from sklearn import metrics

from collections import Counter

import json
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
val_preds, targets = learn.get_preds(DatasetType.Valid)
_ , val_index = val_preds.max(1)
test_preds, y = learn.get_preds(DatasetType.Test)
_ , test_index = test_preds.max(1)
optR = OptimizedRounder()

optR.fit(val_index, targets)

coefficients = optR.coefficients()

val_index = optR.predict(val_index, coefficients)
test_index = optR.predict(test_index, coefficients)

sample_df.diagnosis = test_index.astype(int)

sample_df.to_csv('submission.csv',index=False)
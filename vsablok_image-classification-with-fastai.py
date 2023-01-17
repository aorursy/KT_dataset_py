# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, shutil
for dirname, _, filenames in os.walk('kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#hide
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastai.vision.widgets import *
from fastai.data.all import *
from fastai.vision.core import *
from fastbook import *

#Directory Paths
flower_path = '/kaggle/input/104-flowers-garden-of-eden/jpeg-512x512'
TRAIN_DIR  = flower_path + '/train/'
VAL_DIR  = flower_path + '/val'
TEST_DIR  = flower_path + '/test/'
New_TRAIN_DIR = flower_path + '/train1/'
flowers = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
dls = flowers.dataloaders(TRAIN_DIR, batch_size=32)
dls.train.show_batch(max_n=4, nrows=1)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)

correct = 0
total = len(learn.dls.valid_ds)
correct = (interp.targs == interp.preds.argmax(axis = 1)).sum()
accuracy = correct.float()/total
accuracy
interp.plot_top_losses(6, nrows=6)
#Base model achieved a benchmark of 78.2%. Next we try Augmentations along with Cross Validation set.
flowers = flowers.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = flowers.dataloaders(TRAIN_DIR, batch_size = 32)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)
interp = ClassificationInterpretation.from_learner(learn)

correct = 0
total = len(learn.dls.valid_ds)
correct = (interp.targs == interp.preds.argmax(axis = 1)).sum()
accuracy = correct.float()/total
accuracy
#Data Augmentation gave us significant improvement in accuracy from 78 to 89.37%
test_ids = []
predictions = []
for dirname, _, filenames in os.walk(TEST_DIR):
    for filename in filenames:
          k = plt.imread(os.path.join(dirname,filename))
          pred = learn.predict(k)[1]
          predictions.append(pred)
          test_ids.append(filename.split(".")[0])
# Write the submission file
np.savetxt(
    '/kaggle/working/submission.csv',
    np.rec.fromarrays([test_ids, predictions]),
    fmt=['%s', '%d'],
    delimiter=',',
    header='id,label',
    comments='',
)
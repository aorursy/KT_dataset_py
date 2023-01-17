# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This file contains all the main external libs we'll use

from fastai.imports import * 

import fastai
print(fastai.__version__)


from fastai.utils import *

from fastai.vision import *

from fastai.callbacks import *

from pathlib import Path



import PIL

from torch.utils import model_zoo
PATH = "../input/aerial-cactus-identification/train/"

TMP_PATH = "/tmp/tmp"

MODEL_PATH = "/tmp/model/"

sz=224

train_df = pd.read_csv('../input/aerial-cactus-identification/train.csv')

test_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH)
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot('has_cactus', data=train_df)

plt.title('Classes', fontsize=15)

plt.show()
target_count = train_df.has_cactus.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

# Class count

count_class_0, count_class_1 = train_df.has_cactus.value_counts()

# Divide by class

df_class_0 = train_df[train_df['has_cactus'] == 0]

df_class_1 = train_df[train_df['has_cactus'] == 1]
df_class_0.head()
train_df.info()
X = train_df[:1]

y = train_df['has_cactus']
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(ratio='minority')

# X_sm, y_sm = smote.fit_sample(X, y)

# plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
# from imblearn.over_sampling import (RandomOverSampler, 

#                                     SMOTE, 

#                                     ADASYN)

# sampler = ADASYN(ratio={1: 1927, 0: 300},random_state=0)

# X_rs, y_rs = sampler.fit_sample(X, y)

# print('ADASYN {}'.format(Counter(y_rs)))

# plot_this(X_rs,y_rs)
fnames = np.array([f'train/{f}' for f in sorted(os.listdir(f'{PATH}train'))])

labels = np.array([(0 if 'cat' in fname else 1) for fname in fnames])

fnames[0]
img = plt.imread(f'{PATH}{fnames[10]}')

plt.imshow(img);
img.shape
img[:4,:4]
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)

!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
arch = "../input/resnet34/"
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=20, max_lighting=0.3, max_warp=0.2, max_zoom=1.2)
test_images = ImageList.from_df(test_df, path='../input/aerial-cactus-identification/test', folder='test')

src = (ImageList.from_df(train_df, path='../input/aerial-cactus-identification/train', folder='train')

       .split_by_rand_pct(0.2)

       .label_from_df()

       .add_test(test_images))
data = (src.transform(tfms, 

                     size=32,

                     resize_method=ResizeMethod.PAD, 

                     padding_mode='reflection')

        .databunch(bs=sz)

        .normalize(imagenet_stats))
data.classes, data.c
data.show_batch(rows=4, figsize=(9,9))
Path('models').mkdir(exist_ok=True)

!cp '../input/resnet34/resnet34.pth' 'models/'

def load_url(*args, **kwargs):

    model_dir = Path('models')

    filename  = 'resnet34.pth'

    if not (model_dir/filename).is_file(): raise FileNotFoundError

    return torch.load(model_dir/filename)

model_zoo.load_url = load_url
from google.cloud import bigquery

client = bigquery.Client()


# data = ImageClassifierData.from_names_and_array(

#     path=PATH, 

#     fnames=fnames, 

#     y=labels, 

#     classes=['yes', 'no'], 

#     test_name='test', 

#     tfms=tfms_from_model(arch, sz)

# )

#learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH)

#learn.fit(0.01, 2) 

learn = cnn_learner(data,

                    models.resnet34, 

                    metrics=[accuracy, AUROC()], 

                    path = '.')
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-2

learn.fit_one_cycle(10, lr)
learn.recorder.plot_losses()
learn.save('Model-1')
learn.recorder.plot_lr(show_moms=True)
learn = cnn_learner(data,

                    models.resnet34, 

                    metrics=[accuracy, AUROC()], 

                    callback_fns=[partial(SaveModelCallback)],

                    path = '.')

learn = learn.load('Model-1')
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-6

learn.fit_one_cycle(10, lr)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(2,2))
interp.plot_top_losses(4, figsize=(6,6), heatmap=False)
probability, classification = learn.get_preds(ds_type=DatasetType.Test)

test_df.has_cactus = probability.numpy()[:, 0]

test_df.head()
test_df.to_csv("submission.csv", index=False)
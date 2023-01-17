%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from torchvision.models import * 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os
path = Path("../input/histopathologic-cancer-detection/")

train_labels = pd.read_csv(path/"train_labels.csv")

train_labels.head()
classes = pd.unique(train_labels["label"]);

for i in classes:

    print("{} items in class {}".format(len(train_labels[train_labels["label"] == i]), classes[i]))
transforms = get_transforms(

                do_flip=True, 

                flip_vert=True, 

                max_rotate=10.0, 

                max_zoom=1.1, 

                max_lighting=0.05,

                max_warp=0.2,

                p_affine=0.75,

                p_lighting=0.75

            )
np.random.seed(22)
IMG_SIZE = 32

BATCH_SIZE = 256
data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test', suffix=".tif", size = IMG_SIZE, bs = BATCH_SIZE,

                               ds_tfms = transforms)

data.path = pathlib.Path('.')

stats = data.batch_stats()        

data.normalize(stats)
print(data.classes)

data.c
data.show_batch(rows=5, figsize=(12,9))
arch = models.densenet121

learn = cnn_learner(data, arch, pretrained = True, metrics = [accuracy])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr = slice(1e-5,1e-3))
IMG_SIZE = 64

BATCH_SIZE = 128
data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test', suffix=".tif", size = IMG_SIZE, bs = BATCH_SIZE,

                               ds_tfms = transforms)

data.path = pathlib.Path('.')

stats = data.batch_stats()      

data.normalize(stats)
learn.freeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr = slice(1e-4,1e-3))
from sklearn.metrics import roc_auc_score



def auc_score(y_pred, y_true, tens=True):

    score = roc_auc_score(y_true, torch.sigmoid(y_pred)[:,1])

    if tens:

        score = tensor(score)

    else:

        score = score

    return score
interpretation = ClassificationInterpretation.from_learner(learn)

interpretation.plot_confusion_matrix()
predictions,y = learn.TTA()

acc = accuracy(predictions, y)

print('Final accuracy of the model: {} %.'.format(acc * 100))

prediction_score = auc_score(predictions,y).item()

print('Final AUC of the model: {}.'.format(prediction_score))
submissions = pd.read_csv(path/'sample_submission.csv')

id_list = list(submissions.id)

predictions,y = learn.TTA(ds_type=DatasetType.Test)

prediction_list = list(predictions[:,1])

prediction_dict = dict((key, value.item()) for (key, value) in zip(learn.data.test_ds.items, prediction_list))

prediction_ordered = [prediction_dict[path/('test/' + id + '.tif')] for id in id_list]

submissions = pd.DataFrame({'id':id_list,'label':prediction_ordered})

submissions.to_csv("submission_result.csv",index = False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
from fastai import *

from fastai.vision import *

import torch
data_folder = Path("../input")
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device= torch.device('cuda'))

        .normalize(imagenet_stats)

       )
train_img.show_batch(rows=3, figsize=(7,6))
print(train_img.classes)

print("No. of classes : {}".format(train_img.c))
learner = cnn_learner(train_img, models.densenet161, metrics=[accuracy, error_rate])
learner.lr_find()

learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(5, max_lr=slice(3e-02))
learner.save('stage-1')
interpreter = ClassificationInterpretation.from_learner(learner)
interpreter.plot_confusion_matrix(figsize=(6,6), dpi=60)
interpreter.plot_top_losses(9, figsize=(15,11))
preds,_ = learner.get_preds(ds_type=DatasetType.Test)

test_df.has_cactus = preds.numpy()[:, 0]

test_df.to_csv("submission.csv", index=False)
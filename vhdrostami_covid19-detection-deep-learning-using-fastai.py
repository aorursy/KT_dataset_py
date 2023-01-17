%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pandas as pd

import os

import shutil

from fastai.vision import *

from fastai.metrics import error_rate
# create dataset folder with following subfolders:

#.     NORMAL/

#.     PNEUMONIA/

#.     COVID



dataset_path = './dataset'

!mkdir dataset

!mkdir dataset/NORMAL

!mkdir dataset/PNEUMONIA

!mkdir dataset/COVID
# copy NORMAL and PNEUMONIA xrays from chest-xray-pneumonia repository

!cp ../input/chest-xray-pneumonia/chest_xray/train/NORMAL/* dataset/NORMAL/

!cp ../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/* dataset/PNEUMONIA/
# copy COVID xrays from covid-chest-xray repository



covid_dataset_path = '../input/covid-chest-xray'

csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])

df = pd.read_csv(csvPath)



for (i, row) in df.iterrows():

    if row["finding"] != "COVID-19" or row["view"] != "PA":

        continue



    imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])



    if not os.path.exists(imagePath):

        continue



    filename = row["filename"].split(os.path.sep)[-1]

    outputPath = os.path.sep.join(["dataset/COVID", filename])

    shutil.copy2(imagePath, outputPath)
path = Path("./dataset")

data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), valid_pct=0.2, size=256).normalize(imagenet_stats)
data.show_batch(rows=10, figsize=(10,10))
learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate, Recall(average='macro'), Precision(average='macro')])
learn.fit_one_cycle(4)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(5, min_grad_lr)
learn.save('after_unfreezing')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(6,6), dpi=60)

interp.plot_top_losses(16, figsize=(15,11))
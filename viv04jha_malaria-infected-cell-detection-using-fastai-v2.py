import random, os

import numpy as np

import torch

from fastai.vision.all import *


print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images"))
path = Path("../input/cell-images-for-detecting-malaria/cell_images/cell_images")
path.ls()
fns = get_image_files(path)
fns
failed = verify_images(fns)

failed
cells = DataBlock(

    blocks=(ImageBlock, CategoryBlock), 

    get_items=get_image_files, 

    splitter=RandomSplitter(valid_pct=0.2, seed=42),

    get_y=parent_label,

    item_tfms=Resize(128))
dls = cells.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
cells = cells.new(item_tfms=Resize(128, ResizeMethod.Squish), batch_tfms=aug_transforms(mult=2))

dls = cells.dataloaders(path)

dls.train.show_batch(max_n=8, nrows=2, unique = True)
learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(6)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,8))
interp.plot_top_losses(5, nrows=5)
learn.save('stage-1')
preds, y_true=learn.get_preds()

y_true.shape, preds.shape
y_true=y_true.numpy() 

preds=np.argmax(preds.numpy(), axis=-1)

y_true.shape, preds.shape
from sklearn.metrics import auc, roc_curve, precision_recall_curve, classification_report

classes = list(dls.vocab)

report = classification_report(y_true, preds, target_names=classes)

print(report)
learn.show_results()
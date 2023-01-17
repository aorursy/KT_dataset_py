import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
# Defining the path where my JPG images are stored:
trainpath = "../input/tray-food-segmentation/TrayDataset/TrayDataset/XTrain/"

# Defining the path where the mask PNG images of the images mentioned above are stored:
trainlabel = "../input/tray-food-segmentation/TrayDataset/TrayDataset/yTrain/"

# Taking out the JPG filenames out of the training data folder:
fnames = get_image_files(trainpath)
get_y = lambda x: trainlabel + x.stem + '.png' 
for i in range(0,4):
    img = open_image(fnames[i])
    mask = open_mask(get_y(fnames[i]))
    img.show(figsize=(6,6)), mask.show(figsize=(6,6), alpha=1)
src_shape = np.array(mask.shape[1:])

size = src_shape // 2

bs = 4
classescsv = pd.read_csv("../input/tray-food-segmentation/classes.csv")

classescsv.head()
classes = list(classescsv['_class'])
src = (SegmentationItemList
       .from_folder(trainpath)
       .split_by_rand_pct(.2)
       .label_from_func(get_y, classes=classes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
data.show_batch(1, figsize=(6,6))
learner = unet_learner(data, models.resnet34)
learner.fit_one_cycle(12)
learner.show_results(figsize=(15,15))
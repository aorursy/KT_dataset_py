import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/dataset_updated/"))

%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai import *
from fastai.vision import *
PATH = "../input/dataset_updated/dataset_updated/"
PATH_OLD = "../input/dataset_updated/dataset_updated/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=200
# GPU required
torch.cuda.is_available()
torch.backends.cudnn.enabled
os.listdir(PATH + 'training_set')
files = os.listdir(f'{PATH}training_set/engraving')[:5]
files
img = plt.imread(f'{PATH}training_set/engraving/{files[1]}')
plt.imshow(img);
img.shape
img[:4,:4]
# Fix to enable Resnet to live on Kaggle
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
#arch=resnet34
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(PATH, train='training_set', valid='validation_set', ds_tfms=tfms, size=sz, num_workers=0)
data.show_batch(rows=3, figsize=(6,6))
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir=MODEL_PATH)

learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)
learn.recorder.plot()
learn.recorder.plot_losses()
preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(data, preds, y, losses)
interp.plot_top_losses(9, figsize=(14,14))
interp.plot_confusion_matrix()
interp.most_confused(slice_size=10)
# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# install this version to avoid the multiple warning 
!pip install "torch==1.4" "torchvision==0.5.0"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from albumentations import *
import cv2
import copy
import os
import torch
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt
import glob

#!pip install pretrainedmodels
from tqdm import tqdm_notebook as tqdm
from torchvision.models import *
#import pretrainedmodels

from pathlib import Path
from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
from fastai.callbacks import * 

#from utils import *
import sys

from sklearn.metrics import f1_score, accuracy_score

# Any results you write to the current directory are saved as output.
## set the data folder
data_folder=Path('../input/identify-the-dance-form')

print(os.listdir(data_folder))
recompute_scale_factor=True
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
train_data_path = "../input/identify-the-dance-form/train"
train_path = os.path.join(train_data_path, "*jpg")
test_data_path = "../input/identify-the-dance-form/test"
test_path = os.path.join(test_data_path, "*jpg")
train_files = glob.glob(train_path)
train_images=[]
for file in train_files:
    image = cv2.imread(file)
#     print(image.shape)
    train_images.append(image)
print(len(train_images))
test_files = glob.glob(test_path)
test_images=[]
for file in test_files:
    image = cv2.imread(file)
    print(image.shape)
    test_images.append(image)
print(len(test_images))
## read the csv data files
train_df = pd.read_csv('../input/identify-the-dance-form/train.csv')
test_df = pd.read_csv('../input/identify-the-dance-form/test.csv')
train_df.head(3)
test_df.head(3)
train_df['target']=train_df['target'].map({'mohiniyattam':0,'odissi':1,'kathakali':2,
                                           'bharatanatyam':3,'kuchipudi':4,'sattriya':5,
                                           'kathak':6,'manipuri':7})
train_df.target.value_counts()
##transformations to be done to images

tfms = get_transforms(do_flip=True,flip_vert=False ,max_rotate=10.0, max_zoom=1.22, max_lighting=0.22, max_warp=0.4, p_affine=0.75,
                      p_lighting=0.75)


test_img = ImageList.from_df(test_df, path=data_folder, folder='test')
## create source of train image databunch
np.random.seed(45)

src = (ImageList.from_df(train_df, path=data_folder, folder='train')
       .split_by_rand_pct(0.2)
       #.split_none()
       .label_from_df()
       .add_test(test_img))
# considering image size of 128

data = (src.transform(tfms, size=128,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
        .databunch(path='.', bs=32, device= torch.device('cuda:0')).normalize(imagenet_stats));
print(data.classes)
data.show_batch(rows=3, figsize=(7,7))
# acc_02 = partial(accuracy_thresh, thresh=0.2)
# f_score = partial(fbeta, thresh=0.2)
#lets create learner. tried with resnet152, densenet201, resnet101
# learn = cnn_learner(data=data, base_arch=models.resnet152, metrics=[FBeta(beta=1, average='macro'), accuracy],
#                     callback_fns=ShowGraph).mixup()

# will train first without mixup

#lets create learner. tried with resnet152, densenet201, resnet101
# learn = cnn_learner(data=data, base_arch=models.resnet152, metrics=[FBeta(beta=1, average='macro'), accuracy],
#                     callback_fns=ShowGraph).mixup()

learn = cnn_learner(data=data, base_arch=models.resnet50, metrics=[FBeta(beta=1, average='macro'), accuracy],
                    callback_fns=ShowGraph)


learn.fit_one_cycle(5)
learn.lr_find()
learn.recorder.plot()
# lr=1e-03
learn.fit_one_cycle(10, max_lr=1e-03)

# learn.fit_one_cycle(5, slice(lr))
learn.fit_one_cycle(10, max_lr=1e-04)
learn.save('stage-1-resnet-152-img_size-128')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
interp.plot_top_losses(9, figsize=(15,11))

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
# previously is is trained for 1e-4 
learn.fit_one_cycle(10, slice(1e-4),wd=0.1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
learn.save('stage-2-rn152')
# considering image size of 256
data = (src.transform(tfms, size=256,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
        .databunch(path='.', bs=16, device= torch.device('cuda:0')).normalize(imagenet_stats));

learn.data = data
data.train_ds[0][0].shape
# As in previous layer we unfreeze the whole model so let's freeze it once again so that we will train 
# for last layers only
learn.freeze()
learn.lr_find()
learn.recorder.plot()
lr=8e-5

# lr=3e-06
# model seems to overfit try to use weight decay wd=0.1
learn.fit_one_cycle(10, slice(lr),wd=0.1)
learn.save('stage-1-256-rn152')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
torch.cuda.empty_cache()
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
# lr=1e-05

# lr=1e-04
learn.fit_one_cycle(10, slice(1e-4, lr/5),wd=0.2)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
learn.save('stage-2-256-rn152')
# considering image size of 512
data = (src.transform(tfms, size=512,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
        .databunch(path='.', bs=8, device= torch.device('cuda:0')).normalize(imagenet_stats));

learn.data = data
data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()
learn.recorder.plot()
# lr=1e-03

lr=3e-04
# learn.fit_one_cycle(15, slice(5e-4, lr/5))

learn.fit_one_cycle(10, slice(lr),wd=0.1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
learn.save('stage-1-512-rn152')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
# lr=1e-05
lr=3e-06
learn.fit_one_cycle(10, slice(3e-06, lr/10),wd=0.1)

# In next step what you can try is to run for only 10 epochs to avoid overfitting.
learn.save('stage-2-512-rn152')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
##learn.TTA improves score further. lets see for the validation set
pred_val,y = learn.TTA(ds_type=DatasetType.Valid)
valid_preds = [np.argmax(pred_val[i]) for i in range(len(pred_val))]
valid_preds = np.array(valid_preds)
y = np.array(y)
accuracy_score(valid_preds,y),f1_score(valid_preds,y, average='weighted')
# preds,y = learn.TTA(ds_type=DatasetType.Test)
preds,_ = learn.get_preds(ds_type = DatasetType.Test)
labelled_preds = [np.argmax(preds[i]) for i in range(len(preds))]

labelled_preds = np.array(labelled_preds)
#create submission file
df = pd.DataFrame({'Image':test_df['Image'], 'target':labelled_preds}, columns=['Image', 'target'])

df['target']=df['target'].map({0:'mohiniyattam',1:'odissi',2:'kathakali',
                                           3:'bharatanatyam',4:'kuchipudi',5:'sattriya',
                                           6:'kathak',7:'manipuri'})

df.head()

df.to_csv('submission_mode_resnet-Stage2_512_new.csv', index=False)
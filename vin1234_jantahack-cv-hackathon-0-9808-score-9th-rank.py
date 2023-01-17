# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# from google.colab.patches import cv2_imshow
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from albumentations import *
import cv2
import copy
import os
print(os.listdir("../input"))

#!pip install pretrainedmodels
from tqdm import tqdm_notebook as tqdm
from torchvision.models import *
#import pretrainedmodels

from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
from fastai.callbacks import * 

#from utils import *
import sys

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.metrics import f1_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import glob
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from pathlib import Path
from fastai import *
from fastai.vision import *
import torch
from fastai.callbacks.hooks import *
## set the data folder
data_folder = Path("../input/hp-2020/jh_2020")
print(os.listdir(data_folder))
# ../input/hp-2020/jh_2020/images
print(os.listdir('../input/hp-2020/jh_2020/images'))
data_path = "../input/hp-2020/jh_2020/images/"
path = os.path.join(data_path , "*jpg")

print(os.listdir(data_path))
files = glob.glob(path)
data=[]
for file in files:
    image = cv2.imread(file)
    data.append(image)
# data[1]
## read the csv data files
train_df = pd.read_csv('../input/hp-2020/jh_2020/train.csv')
test_df = pd.read_csv('../input/hp-2020/jh_2020/test.csv')
submit = pd.read_csv('../input/hp-2020/jh_2020/sample_submission.csv')
print(test_df.shape)
print(train_df.shape)
train_images = data[:1646]
test_images= data[1646:]
train_df.head(2)
train_df['emergency_or_not'].value_counts()
dupli=train_df[train_df['emergency_or_not']==1].sample(150)
dupli.shape
dupli.head(4)
# path="/content/images"
print(train_df.shape)

# for i in dupli['image_names']:    
#     new_image='copy'+i
#     original_img = cv2.imread("/content/images/"+i)
# #     clone_img = copy.copy(original_img)
# #     data.append(clone_img)
#     cv2.imwrite(data_path + new_image,original_img)

#     d=[{'image_names':new_image,'emergency_or_not':1}]

#     train_df=train_df.append(d,ignore_index=True,sort=False)
# print(train_df.shape)
# len(listdir(path))
train_df['emergency_or_not'].value_counts()

##transformations to be done to images
tfms = get_transforms(do_flip=False,flip_vert=False ,max_rotate=10.0, max_zoom=1.22, max_lighting=0.22, max_warp=0.4, p_affine=0.75,
                      p_lighting=0.75)
#, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))


#Apply new transformations

# tfms = get_transforms(do_flip=False,flip_vert=False ,max_rotate=10.0, max_zoom=1.22, max_lighting=0.22, max_warp=0.4, p_affine=0.75,
#                       p_lighting=0.75, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))

## create databunch of test set to be passed
test_img = ImageList.from_df(test_df, path=data_folder, folder='images')
np.random.seed(45)
## create source of train image databunch
src = (ImageList.from_df(train_df, path=data_folder, folder='images')
       .split_by_rand_pct(0.2)
       #.split_none()
       .label_from_df()
       .add_test(test_img))
data = (src.transform(tfms, size=300,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
        .databunch(path='.', bs=16, device= torch.device('cuda:0')).normalize(imagenet_stats))

# data = (src.transform(tfms, size=350,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)
#         .databunch(path='.', bs=16).normalize(imagenet_stats))
print(data.classes)
data.show_batch(rows=3, figsize=(7,7))
#lets create learner. tried with resnet152, densenet201, resnet101
# learn = cnn_learner(data=data, base_arch=models.densenet201, metrics=[FBeta(beta=1, average='macro'),error_rate],
#                     callback_fns=ShowGraph)



learn = cnn_learner(data=data, base_arch=models.densenet201, metrics=[FBeta(beta=1, average='macro'), accuracy],
                    callback_fns=ShowGraph).mixup()
# learn.summary()

#lets find the correct learning rate to be used from lr finder
learn.lr_find()
learn.recorder.plot(suggestion=True)
#lets start with steepset slope point. adding wd (weight decay) not to overfit as we are running 10 epochs 
lr = 1e-04
#learn.fit_one_cycle(10, slice(lr))
# learn.fit_one_cycle(10, max_lr=slice(lr), wd=0.2)

# learn.fit_one_cycle(5, max_lr=slice(3e-5,3e-2))         # previous


# learn.fit_one_cycle(5, max_lr=slice(3e-5,1e-3),wd=0.2)   # with this we got better accuracy 

# for first layer don't use slice as it used to train other layers.

learn.fit_one_cycle(10, slice(lr),wd=0.1)   
learn.fit_one_cycle(10, slice(lr),wd=0.1) 
# save the model

learn.save('stage1_model')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
#lets plot the lr finder record
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
# train for  more cycles after unfreezing
# learn.fit_one_cycle(5,slice(1e-05),wd=0.15)
learn.fit_one_cycle(10, slice(1e-03, lr/10),wd=0.15)
# save the model

learn.save('stage2_model')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
learn.freeze_to(-3)
## finding the LR
learn.lr_find()
learn.recorder.plot(suggestion=True)
# 96.52% accuracy achieved
learn.fit_one_cycle(6, slice(1e-02, lr/10),wd=0.15)
# save the model

learn.save('stage3_model')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
## freezing initial all layers except last 2 layers

# 96.18%
learn.freeze_to(-2)
## finding the LR
learn.lr_find()
learn.recorder.plot(suggestion=True)
## training for few cylcles more
learn.fit_one_cycle(5, slice(5e-04, lr/40),wd=0.15)
## training for few cylcles more
learn.fit_one_cycle(5, slice(5e-04, lr/100),wd=0.15)
# save the model

learn.save('stage4_model_new')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(dpi=120)
learn.freeze_to(-1)
## finding the LR
learn.lr_find()
learn.recorder.plot(suggestion=True)
## training even more
learn.fit_one_cycle(5, slice(1e-05, lr/30),wd=0.05)
learn.fit_one_cycle(6, slice(5e-06, lr/100))

##learn.TTA improves score further. lets see for the validation set
pred_val,y = learn.TTA(ds_type=DatasetType.Valid)
from sklearn.metrics import f1_score, accuracy_score
valid_preds = [np.argmax(pred_val[i]) for i in range(len(pred_val))]
valid_preds = np.array(valid_preds)
y = np.array(y)
accuracy_score(valid_preds,y),f1_score(valid_preds,y, average='micro')

# y
# preds,y = learn.TTA(ds_type=DatasetType.Test)
preds,_ = learn.get_preds(ds_type = DatasetType.Test)
labelled_preds = [np.argmax(preds[i]) for i in range(len(preds))]

labelled_preds = np.array(labelled_preds)
# test_df.shape
len(labelled_preds)
#create submission file
df = pd.DataFrame({'image_names':test_df['image_names'], 'emergency_or_not':labelled_preds}, columns=['image_names', 'emergency_or_not'])
df.to_csv('submission_model_new1.csv', index=False)
d=pd.read_csv('submission1.csv')
d.head()
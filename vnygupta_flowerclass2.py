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

data_folder = Path("../input/flowerdata/he_challenge_data/data/")
data_path = "../input/flowerdata/he_challenge_data/data/train/"

path = os.path.join(data_path , "*jpg")
path
# files = glob.glob(path)

# data=[]

# for file in files:

#     image = cv2.imread(file)

#     data.append(image)
## read the csv data files

train_df = pd.read_csv('../input/flowerdata/he_challenge_data/data/train.csv')

test_df = pd.read_csv('../input/flowerdata/he_challenge_data/data/test.csv')

submit = pd.read_csv('../input/flowerdata/he_challenge_data/data/sample_submission.csv')
train_df.shape, test_df.shape
train_df.head()
train_df.image_id=train_df.image_id.apply(lambda x:str(x)+str('.jpg'))
test_df.image_id=test_df.image_id.apply(lambda x:str(x)+str('.jpg'))
# train_df.groupby('category').count()
##transformations to be done to images

tfms = get_transforms(do_flip=True,flip_vert=True ,max_rotate=40.0, max_zoom=1.22, max_lighting=0.22, max_warp=0.0, p_affine=0.75,

                      p_lighting=0.75)

#, xtra_tfms=zoom_crop(scale=(0.9,1.8), do_rand=True, p=0.8))



## create databunch of test set to be passed

test_img = ImageList.from_df(test_df, path=data_folder, folder='test')
np.random.seed(145)

## create source of train image databunch

src = (ImageList.from_df(train_df, path=data_folder, folder='train')

       .split_none()

       #.split_none()

       .label_from_df()

       .add_test(test_img))
data = (src.transform(tfms, size=299,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)

        .databunch(path='.', bs=32, device= torch.device('cuda:0')).normalize(imagenet_stats))



# data = (src.transform(tfms, size=484,padding_mode='reflection',resize_method=ResizeMethod.SQUISH)

#         .databunch(path='.', bs=16, device= torch.device('cuda:0')).normalize(imagenet_stats))
## lets see the few images from our databunch

#data.show_batch(rows=3, figsize=(12,12))
#lets create learner. tried with resnet152, densenet201, resnet101

learn = cnn_learner(data=data, base_arch=models.resnet101, metrics=[FBeta(beta=1, average='macro'), accuracy],

                    callback_fns=ShowGraph)



# learn = cnn_learner(data=data, base_arch=models.densenet161, metrics=[FBeta(beta=1, average='macro'), accuracy],

#                     callback_fns=ShowGraph).mixup()
#lets find the correct learning rate to be used from lr finder

learn.lr_find()

learn.recorder.plot(suggestion=True)
#lets start with steepset slope point. adding wd (weight decay) not to overfit as we are running 15 epochs 

lr = 1e-03

#learn.fit_one_cycle(10, slice(lr))

learn.fit_one_cycle(25, slice(lr), wd=0.2)
# ##learn.TTA improves score further. lets see for the validation set

# pred_val,y = learn.TTA(ds_type=DatasetType.Valid)

# from sklearn.metrics import f1_score, accuracy_score

# valid_preds = [np.argmax(pred_val[i])+1 for i in range(len(pred_val))]

# valid_preds = np.array(valid_preds)

# y = np.array(y+1)

# accuracy_score(valid_preds,y),f1_score(valid_preds,y, average='micro')
preds,_ = learn.TTA(ds_type=DatasetType.Test)

#preds,_ = learn.get_preds(ds_type = DatasetType.Test)

labelled_preds = [np.argmax(preds[i])+1 for i in range(len(preds))]



labelled_preds = np.array(labelled_preds)
#create submission file

df = pd.DataFrame({'image_id':test_df['image_id'], 'category':labelled_preds}, columns=['image_id', 'category'])

df.to_csv('submission18sepv1.csv', index=False)
## function to create download link

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
create_download_link(filename = 'submission18sepv1.csv')
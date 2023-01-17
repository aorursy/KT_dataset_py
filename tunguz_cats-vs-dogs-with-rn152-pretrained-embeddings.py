# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import log_loss, roc_auc_score

from sklearn.linear_model import LogisticRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample_submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

sample_submission.tail()
sample_submission.head()
TRAIN_FOLDER = '../input/cats-and-dogs-embedded-data/train/train/'

TEST_FOLDER =  '../input/cats-and-dogs-embedded-data/test/test/'

IMG_SIZE = 224
train_image_list = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_image_list.npy')

test_image_list = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/test_image_list.npy')
train_image_list
test_image_list
ids = [int(x.split('.')[0]) for x in test_image_list]

ids[:20]
target = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/target.npy')
target
train_ResNet152 = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_2/train_ResNet152.npy')

test_ResNet152 = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_2/test_ResNet152.npy')
train_ResNet152.shape
X_train, X_val, y_train, y_val = train_test_split(train_ResNet152, target, test_size=0.1, random_state=42)
%%time

lr = LogisticRegression(C=0.2, max_iter=10000)

lr.fit(X_train, y_train)
val_preds = lr.predict_proba(X_val)[:,1]
val_preds
y_val
roc_auc_score(y_val, val_preds)
log_loss(y_val, val_preds)
test_preds = lr.predict_proba(test_ResNet152)[:,1]
test_preds
sample_submission['id'] = ids

sample_submission['label'] = test_preds

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()
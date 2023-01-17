!pip install --use-feature=2020-resolver https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.3.0_SNAPSHOT%2B00b0ad1293b4fa74d6aca5da4e9ab7a9d16777f0-py3-none-manylinux2010_x86_64.whl
import xgboost as xgb

xgb.__version__
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import log_loss, roc_auc_score

import gc



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
train_EB7 = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_3/cats_and_dogs_3/train_EB7.npy')

test_EB7 = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_3/cats_and_dogs_3/test_EB7.npy')
train_EB7.shape
X_train, X_val, y_train, y_val = train_test_split(train_EB7, target, test_size=0.1, random_state=42)
xgb_params = {

    'eta': 0.05,

    'max_depth': 4,

    'subsample': 0.8,

    'colsample_bytree': 0.6,

    'alpha': 0.01,

    'lambda': 1.00,

    'gamma' : 0.02,

    'max_bin': 256,

    'objective': 'reg:logistic',

    'eval_metric': 'auc',

    'verbosity': 0,

    'tree_method': 'gpu_hist', 

    'predictor': 'gpu_predictor'

}



dtrain = xgb.DMatrix(X_train, y_train)

dval = xgb.DMatrix(X_val, y_val)

dtest = xgb.DMatrix(test_EB7)
del train_EB7, test_EB7, X_train, X_val

gc.collect()
%%time

model = xgb.train(xgb_params, dtrain, num_boost_round=106)
val_preds = model.predict(dval)
val_preds
y_val
roc_auc_score(y_val, val_preds)
log_loss(y_val, val_preds)
%%time

val_shap_preds = model.predict(dval, pred_contribs=True)
%%time

test_shap_preds = model.predict(dtest, pred_contribs=True)



%%time

train_shap_preds = model.predict(dtrain, pred_contribs=True)
test_preds = model.predict(dtest)
test_preds.shape
sample_submission['id'] = ids

sample_submission['label'] = test_preds

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()
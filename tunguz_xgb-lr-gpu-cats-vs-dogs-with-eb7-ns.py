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

from sklearn.linear_model import LogisticRegression

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
train_EB7_ns = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/train_EB7_ns.npy')

test_EB7_ns = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/test_EB7_ns.npy')

train_EB4_ns = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/train_EB4_ns.npy')

test_EB4_ns = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/test_EB4_ns.npy')
train_EB7_ns.shape
X_train, X_val, y_train, y_val = train_test_split(train_EB7_ns, target, test_size=0.1, random_state=42)
lr = LogisticRegression(C=0.026, max_iter=10000)

lr.fit(X_train, y_train)

val_preds_EB7_lr = lr.predict_proba(X_val)[:,1]

test_preds_EB7_lr = lr.predict_proba(test_EB7_ns)[:,1]

print(roc_auc_score(y_val, val_preds_EB7_lr))

print(log_loss(y_val, val_preds_EB7_lr))
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

dtest = xgb.DMatrix(test_EB7_ns)
del train_EB7_ns, test_EB7_ns, X_train, X_val

gc.collect()
%%time



model = xgb.train(xgb_params, dtrain, num_boost_round=100)

val_preds_xgb_1 = model.predict(dval)

test_preds_xgb_1 = model.predict(dtest)
roc_auc_score(y_val, val_preds_xgb_1)
0.9999903999447037
log_loss(y_val, val_preds_xgb_1)
0.008699688403122127
roc_auc_score(y_val, 0.5*val_preds_xgb_1+0.5*val_preds_EB7_lr)
log_loss(y_val, 0.5*val_preds_xgb_1+0.5*val_preds_EB7_lr)
#%%time

#val_shap_preds = model.predict(dval, pred_contribs=True)
#%%time

#test_shap_preds = model.predict(dtest, pred_contribs=True)



#%%time

#train_shap_preds = model.predict(dtrain, pred_contribs=True)
test_preds_xgb_1.shape
X_train, X_val, y_train, y_val = train_test_split(train_EB4_ns, target, test_size=0.1, random_state=42)
lr = LogisticRegression(C=0.019, max_iter=10000)

lr.fit(X_train, y_train)

val_preds_EB4_lr = lr.predict_proba(X_val)[:,1]

test_preds_EB4_lr = lr.predict_proba(test_EB4_ns)[:,1]

print(roc_auc_score(y_val, val_preds_EB4_lr))

print(log_loss(y_val, val_preds_EB4_lr))
xgb_params = {

    'eta': 0.05,

    'max_depth': 4,

    'subsample': 0.85,

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

dtest = xgb.DMatrix(test_EB4_ns)
del train_EB4_ns, test_EB4_ns, X_train, X_val

gc.collect()
%%time

val_preds_xgb_2 = 0

test_preds_xgb_2 = 0



n_seed = 10



for j in range(n_seed):

    xgb_params['seed'] = 3*j**2+154

    model = xgb.train(xgb_params, dtrain, num_boost_round=100)

    val_preds_xgb_2 += model.predict(dval)/n_seed

    test_preds_xgb_2 += model.predict(dtest)/n_seed
sample_submission['id'] = ids

sample_submission['label'] = test_preds_xgb_1

sample_submission.to_csv('submission_xgb_1.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = test_preds_xgb_2

sample_submission.to_csv('submission_xgb_2.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = 0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr

sample_submission.to_csv('submission_0.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = (0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)**0.99

sample_submission.to_csv('submission_0_b.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = (0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)**0.95

sample_submission.to_csv('submission_0_c.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = 0.47*test_preds_xgb_1+0.53*test_preds_EB7_lr

sample_submission.to_csv('submission_1.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = 0.52*test_preds_xgb_1+0.48*test_preds_EB7_lr

sample_submission.to_csv('submission_2.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = 0.55*test_preds_xgb_1+0.45*test_preds_EB7_lr

sample_submission.to_csv('submission_3.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = 0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr

sample_submission.to_csv('submission_EB4_0.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = np.clip(1.01*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.01*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

sample_submission.to_csv('submission_EB7_EB4_0.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = np.clip(1.05*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.05*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

sample_submission.to_csv('submission_EB7_EB4_1.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = np.clip(1.08*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.08*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

sample_submission.to_csv('submission_EB7_EB4_2.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = np.clip(1.08*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.08*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

sample_submission.to_csv('submission_EB7_EB4_2.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

subs = np.clip(1.1*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.1*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

sample_submission['label'] = subs

sample_submission.to_csv('submission_EB7_EB4_3.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

subs = np.clip(1.1*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.1*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

subs[subs > 0.9976] = 1

sample_submission['label'] = subs

sample_submission.to_csv('submission_EB7_EB4_3_b.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

subs = np.clip(1.1*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.1*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

subs[subs > 0.9973] = 1

sample_submission['label'] = subs

sample_submission.to_csv('submission_EB7_EB4_3_c.csv', index=False)

sample_submission.head()
subs[subs < 0.0018].shape
sample_submission['id'] = ids

subs = np.clip(1.1*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.1*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

subs[subs > 0.9973] = 1

subs[subs <  0.0018] = 0

sample_submission['label'] = subs

sample_submission.to_csv('submission_EB7_EB4_3_d.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = np.clip(1.12*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.12*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)

sample_submission.to_csv('submission_EB7_EB4_4.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

subs = np.clip(1.12*(0.5*test_preds_xgb_1+0.5*test_preds_EB7_lr)-0.12*(0.5*test_preds_xgb_2+0.5*test_preds_EB4_lr), 0, 1)



sample_submission['label'] = subs

subs[subs > 0.9974] = 1

subs[subs <  0.0018] = 0

sample_submission.to_csv('submission_EB7_EB4_4_b.csv', index=False)

sample_submission.head()
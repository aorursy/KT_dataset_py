import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
!pip install --use-feature=2020-resolver https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.3.0_SNAPSHOT%2B00b0ad1293b4fa74d6aca5da4e9ab7a9d16777f0-py3-none-manylinux2010_x86_64.whl

import xgboost as xgb

xgb.__version__
import numpy as np

import pandas as pd

import cudf

import cupy as cp

from cuml.neighbors import KNeighborsClassifier

from cuml.linear_model import LogisticRegression

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_auc_score, log_loss
sample_submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

sample_submission.tail()
train_image_list = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/train_image_list.npy')

test_image_list = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/test_image_list.npy')
ids = [int(x.split('.')[0]) for x in test_image_list]
target = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_1/target.npy')
train_EB7_ns = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/train_EB7_ns.npy')

test_EB7_ns = np.load('../input/cats-and-dogs-embedded-data/cats_and_dogs_4/cats_and_dogs_4/test_EB7_ns.npy')
y_oof = np.zeros(train_EB7_ns.shape[0])

y_test = 0
NUM_FOLDS = 20

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=137)
%%time

for f, (train_ind, val_ind) in enumerate(kf.split(train_EB7_ns, target)):

    print(f)

    X_train, X_val = train_EB7_ns[train_ind], train_EB7_ns[val_ind]

    y_train, y_val = target[train_ind], target[val_ind]

    lr = LogisticRegression(C=0.026, max_iter=10000)

    lr.fit(X_train, y_train)

    val_preds = lr.predict_proba(X_val)[:,1]

    test_preds = lr.predict_proba(test_EB7_ns)[:,1]

    y_oof[val_ind] = val_preds

    y_test += test_preds/NUM_FOLDS
roc_auc_score(target, y_oof)
log_loss(target, y_oof)
y_oof_xgb_1 = np.zeros(train_EB7_ns.shape[0])

y_test_xgb_1 = 0



%%time



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



dtest = xgb.DMatrix(test_EB7_ns)



for f, (train_ind, val_ind) in enumerate(kf.split(train_EB7_ns, target)):

    print(f)

    X_train, X_val = train_EB7_ns[train_ind], train_EB7_ns[val_ind]

    y_train, y_val = target[train_ind], target[val_ind]

    

    dtrain = xgb.DMatrix(X_train, y_train)

    dval = xgb.DMatrix(X_val, y_val)

    

    clf = xgb.train(xgb_params, dtrain, num_boost_round=100)

    val_preds = clf.predict(dval)

    test_preds = clf.predict(dtest)

    y_oof_xgb_1[val_ind] = val_preds

    y_test_xgb_1 += test_preds/NUM_FOLDS

    

print(roc_auc_score(target, y_oof_xgb_1))

print(log_loss(target, y_oof_xgb_1))
%%time



y_oof_knn = np.zeros(train_EB7_ns.shape[0])

y_test_knn = 0



kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=137)



for f, (train_ind, val_ind) in enumerate(kf.split(train_EB7_ns, target)):

    print(f)

    X_train, X_val = train_EB7_ns[train_ind], train_EB7_ns[val_ind]

    y_train, y_val = target[train_ind], target[val_ind]

    knc = KNeighborsClassifier(n_neighbors=400)

    knc.fit(X_train, y_train)

    val_preds = knc.predict_proba(X_val)[:,1]

    test_preds = knc.predict_proba(test_EB7_ns)[:,1]

    y_oof_knn[val_ind] = val_preds

    y_test_knn += test_preds/NUM_FOLDS

    

print(roc_auc_score(target, y_oof_knn))

print(log_loss(target, y_oof_knn))
print(roc_auc_score(target, 0.9*y_oof+0.1*y_oof_knn))

print(log_loss(target, 0.9*y_oof+0.1*y_oof_knn))
sample_submission['id'] = ids

sample_submission['label'] = y_test

sample_submission.to_csv('submission_lr.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test_xgb_1

sample_submission.to_csv('submission_xgb_1.csv', index=False)

sample_submission.head()
sample_submission['label'] = y_test_knn

sample_submission.to_csv('submission_knn.csv', index=False)

sample_submission.head()
sample_submission['label'] = 0.9*y_test + 0.1* y_test_knn

sample_submission.to_csv('submission_blend_1.csv', index=False)

sample_submission.head()
sample_submission['label'] = 0.95*y_test + 0.05* y_test_knn

sample_submission.to_csv('submission_blend_2.csv', index=False)

sample_submission.head()
sample_submission['label'] = 0.99*y_test + 0.01* y_test_knn

sample_submission.to_csv('submission_blend_3.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test**1.001

sample_submission.to_csv('submission_lr_1.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test**1.005

sample_submission.to_csv('submission_lr_2.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test**1.01

sample_submission.to_csv('submission_lr_3.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test**0.999

sample_submission.to_csv('submission_lr_4.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test**0.995

sample_submission.to_csv('submission_lr_5.csv', index=False)

sample_submission.head()
sample_submission['id'] = ids

sample_submission['label'] = y_test**0.99

sample_submission.to_csv('submission_lr_6.csv', index=False)

sample_submission.head()
sample_submission['label'] = 0.5*y_test_xgb_1 + 0.5* y_test

sample_submission.to_csv('submission_blend_4.csv', index=False)

sample_submission.head()
sample_submission['label'] = (0.5*y_test_xgb_1 + 0.5* y_test)**0.99

sample_submission.to_csv('submission_blend_5.csv', index=False)

sample_submission.head()
sample_submission['label'] = (0.5*y_test_xgb_1 + 0.5* y_test)**1.01

sample_submission.to_csv('submission_blend_6.csv', index=False)

sample_submission.head()
import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import numpy as np

# import pandas as pd

import cudf

import cupy as cp

from cuml.neighbors import KNeighborsRegressor

from cuml import SVR

from cuml.linear_model import Ridge, Lasso

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from cuml.metrics import mean_absolute_error, mean_squared_error
train = cudf.read_csv("../input/siim-isic-melanoma-classification/train.csv")

test = cudf.read_csv("../input/siim-isic-melanoma-classification/test.csv")
x_train_32 = cp.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')

x_test_32 = cp.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')
x_train_32 = x_train_32.reshape((x_train_32.shape[0], 32*32*3))

x_train_32.shape
x_test_32 = x_test_32.reshape((x_test_32.shape[0], 32*32*3))

x_test_32.shape
NUM_FOLDS = 5

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
y_oof = cp.zeros(train.shape[0])

y_test = cp.zeros(test.shape[0])
y = train['target'].values.reshape(-1,1)
for f, (train_ind, val_ind) in enumerate(kf.split(train, train)):

    print(f)

    train_, val_ = x_train_32[train_ind].astype('float32'), x_train_32[val_ind].astype('float32')

    y_tr, y_vl = y[train_ind].astype('float32'), y[val_ind].astype('float32')

    

        

    model = SVR(C=0.2, cache_size=5000.0)

    model.fit(train_, y_tr)

    

    val_pred = model.predict(val_)

    y_oof[val_ind] = val_pred

    

    y_test += model.predict(x_test_32.astype('float32')).values/NUM_FOLDS

    

    print("Fold AUC:", roc_auc_score(cp.asnumpy(y_vl.flatten()), cp.asnumpy(val_pred.values)))

    

    





print("Total AUC:", roc_auc_score(cp.asnumpy(y.flatten()), cp.asnumpy(y_oof)))
sample_submission = cudf.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission.head()
sample_submission['target'] = y_test

sample_submission.to_csv('submission_32x32_svr.csv', index=False)
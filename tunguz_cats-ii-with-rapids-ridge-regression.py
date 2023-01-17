%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, roc_auc_score

import cudf, cuml

import cupy as cp

from cuml.linear_model import LogisticRegression

import numpy as np

#from cuml.metrics import accuracy_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from cuml.linear_model import Ridge

train = cudf.read_csv('../input/multi-cat-encodings/X_train_te.csv')

test = cudf.read_csv('../input/multi-cat-encodings/X_test_te.csv')

sample_submission = cudf.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')
train_oof = cp.zeros((train.shape[0],))

test_preds = 0

train_oof.shape
def auc_cp(y_true,y_pred):

    y_true = y_true.astype('float32')

    ids = np.argsort(-y_pred) # we want descedning order

    y_true = y_true[ids.values]

    y_pred = y_pred[ids.values]

    zero = 1 - y_true

    acc_one = cp.cumsum(y_true)

    acc_zero = cp.cumsum(zero)

    sum_one = cp.sum(y_true)

    sum_zero = cp.sum(zero)

    tpr = acc_one/sum_one

    fpr = acc_zero/sum_zero

    return calculate_area(fpr,tpr)



def calculate_area(fpr,tpr):

    return cp.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1]))/2
features = test.columns

%%time

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)

scores = []



for jj, (train_index, val_index) in enumerate(kf.split(train)):

    print("Fitting fold", jj+1)

    train_features = train.loc[train['fold_column'] != jj][features]

    train_target = train.loc[train['fold_column'] != jj]['target'].values.astype(float)

    

    val_features = train.loc[train['fold_column'] == jj][features]

    val_target = train.loc[train['fold_column'] == jj]['target'].values.astype(float)

    

    model = Ridge(alpha = 5)

    model.fit(train_features, train_target)

    val_pred = model.predict(val_features)

    train_oof[val_index] = val_pred

    val_target = cp.asarray(val_target)

    score = auc_cp(val_target, val_pred)

    print("Fold AUC:", score)

    scores.append(cp.asnumpy(score))

    test_preds += model.predict(test).values/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()

    

print("Mean AUC:", np.mean(scores))
sample_submission['target'] = test_preds

sample_submission.to_csv('submission.csv', index=False)
cp.save('test_preds', test_preds)

cp.save('train_oof', train_oof)
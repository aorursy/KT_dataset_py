%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cupy as cp # linear algebra

import cudf as cd # data processing, CSV file I/O (e.g. pd.read_csv)



from cuml.svm import SVR

from cuml.decomposition import PCA



from sklearn.model_selection import KFold

from cuml.metrics import accuracy_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = cd.read_csv('../input/digit-recognizer/train.csv')

test = cd.read_csv('../input/digit-recognizer/test.csv')

submission = cd.read_csv('../input/digit-recognizer/sample_submission.csv')
%%time

X = train[train.columns[1:]].values.astype('float32')

test = test.values.astype('float32')

Y = train.label.values.astype('float32')

train_oof = cp.zeros((X.shape[0], ))

test_preds = 0

train_oof.shape
%%time

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)



for jj, (train_index, val_index) in enumerate(kf.split(X)):

    print("Fitting fold", jj+1)

    train_features = X[train_index]

    train_target = Y[train_index]

    val_features = X[val_index]

    val_target = Y[val_index]

    

    model =  SVR(kernel='rbf',C=20)

    model.fit(train_features, train_target)

    val_pred = model.predict(val_features)

    val_pred = cp.clip(val_pred.values, 0, 9)

    train_oof[val_index] = val_pred.astype('int')

    print("Fold accuracy:", accuracy_score(val_target, val_pred.astype('int')))

    test_preds += model.predict(test)/n_splits

          

    del train_features, train_target, val_features, val_target

    gc.collect()
test_preds = cp.clip(test_preds.values, 0, 9)



submission['Label'] = test_preds.astype('int')

submission.to_csv('submission.csv', index=False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, roc_auc_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = np.load('../input/multi-cat-encodings/X_train_le.npy')

test = np.load('../input/multi-cat-encodings/X_test_le.npy')

sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

target = np.load('../input/multi-cat-encodings/target.npy')
np.unique(target)
sample_submission.head()
train_oof = np.zeros((train.shape[0],))

test_preds = 0

train_oof.shape
%%time

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)



for jj, (train_index, val_index) in enumerate(kf.split(train)):

    print("Fitting fold", jj+1)

    train_features = train[train_index]

    train_target = target[train_index]

    

    val_features = train[val_index]

    val_target = target[val_index]

    

    model = HistGradientBoostingClassifier(max_iter=10000, learning_rate=0.01)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)

    train_oof[val_index] = val_pred[:,1]

    print("Fold AUC:", roc_auc_score(val_target, val_pred[:,1]))

    test_preds += model.predict_proba(test)[:,1]/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(roc_auc_score(target, train_oof))

sample_submission['target'] = test_preds

sample_submission.to_csv('submission.csv', index=False)
np.save('test_preds', test_preds)

np.save('train_oof', train_oof)
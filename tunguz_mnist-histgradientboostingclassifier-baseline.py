# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
%%time

X = train[train.columns[1:]].values

Y = train.label.values
train_oof = np.zeros((X.shape[0], 10))

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

    

    model = HistGradientBoostingClassifier(max_iter=400)

    model.fit(train_features, train_target)

    val_pred = model.predict_proba(val_features)

    train_oof[val_index] = val_pred

    print("Fold accuracy:", accuracy_score(val_target, np.argmax(val_pred, axis=1)))

    test_preds += model.predict_proba(test)/n_splits

    del train_features, train_target, val_features, val_target

    gc.collect()
print(accuracy_score(Y, np.argmax(train_oof, axis=1)))
preds = np.argmax(test_preds, axis=1)

preds
submission['Label'] = preds

submission.to_csv('submission.csv', index=False)
np.save('test_preds', test_preds)

np.save('train_oof', train_oof)
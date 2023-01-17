# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.svm import SVC

from sklearn.decomposition import PCA



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
x_train_32 = np.load('../input/siimisic-melanoma-resized-images/x_train_32.npy')

x_test_32 = np.load('../input/siimisic-melanoma-resized-images/x_test_32.npy')
x_train_32 = x_train_32.reshape((x_train_32.shape[0], 32*32*3))

x_train_32.shape
x_test_32 = x_test_32.reshape((x_test_32.shape[0], 32*32*3))

x_test_32.shape
%%time

pca = PCA(n_components=0.99,whiten=True)

x_train_32 = pca.fit_transform(x_train_32)

x_test_32 = pca.transform(x_test_32)

print(x_train_32.shape)

print(x_test_32.shape)
NUM_FOLDS = 5

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
y_oof = np.zeros(train.shape[0])

y_test = np.zeros(test.shape[0])
y = train['target'].values

%%time

for f, (train_ind, val_ind) in enumerate(kf.split(train, train)):

    print(f)

    train_, val_ = x_train_32[train_ind].astype('float32'), x_train_32[val_ind].astype('float32')

    y_tr, y_vl = y[train_ind].astype('float32'), y[val_ind].astype('float32')

    

        

    model = SVC(kernel='rbf',C=1, probability=True)

    model.fit(train_, y_tr)

    

    val_pred = model.predict_proba(val_)[:,1]

    y_oof[val_ind] = val_pred

    

    y_test += model.predict_proba(x_test_32.astype('float32'))[:,1]/NUM_FOLDS

    

    print("Fold AUC:", roc_auc_score(y_vl, val_pred))

    

    





print("Total AUC:", roc_auc_score(y, y_oof))
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission['target'] = y_test

sample_submission.to_csv('submission_32x32_svc.csv', index=False)

sample_submission.head()
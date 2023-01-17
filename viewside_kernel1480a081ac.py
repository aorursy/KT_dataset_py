# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

submission = pd.read_csv('../input/sample_submission.csv')

test = pd.read_csv('../input/test.csv')
train.head()
# x = train.drop(['ID_code','target'],axis=1)[:10000]

# y = train['target'][:10000]

x = train.drop(['ID_code','target'],axis=1)

y = train['target']
# pca=PCA(n_components=0.95)
# x_pca=pca.fit_transform(x)

# x_pca =pd.DataFrame(x_pca)
x.shape
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import time

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn import metrics

from lightgbm import LGBMClassifier
params={'learning_rate': 0.01,

        'objective':'binary',

        'metric':'auc',

        'num_leaves': 31,

        'verbose': 1,

        'random_state':42,

        'bagging_fraction': 1,

        'feature_fraction': 0.7

       }
# model = GaussianNB()

# model = RandomForestClassifier(max_depth=8,n_estimators=1200)

# model = xgb.XGBClassifier(max_depth=7,n_estimators=800,subsample=0.8,colsample_bytree=0.8)

model =LGBMClassifier(**params,n_estimators=10000,n_jobs=-1,max_depth=5)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=1)
model.fit(x_train,y_train,eval_set=[(x_test, y_test)], early_stopping_rounds=500, verbose=200)

# roc_auc_score(y_test,model.predict_proba(x_test)[:,1])

# cross_val_score(model,x,y,cv=5,scoring='roc_auc').mean()
test_pred_ = model.predict_proba(test.drop(['ID_code'],axis=1,),num_iteration=model.best_iteration_)
test_pred_
# kfold = KFold(n_splits=5,random_state=3,shuffle=True)
# test_probs =[]8342

# for fold ,(_train_idx ,_val_idx) in enumerate (kfold.split(x,y)):

#     print("####################################")

#     print("############fold:",fold)

#     x_train = x.iloc[_train_idx]

#     y_train = y.iloc[_train_idx]

#     x_val = x.iloc[_val_idx]

#     y_val = y.iloc[_val_idx]

#     f_time = time.time()

#     model.fit(x_train,y_train)

#     print('cost time %s s'%(time.time()-f_time))

#     print('train_roc:%s'%(roc_auc_score(y_train,model.predict_proba(x_train)[:,1])))

#     print('val_roc:%s'%(roc_auc_score(y_val,model.predict_proba(x_val)[:,1])))

#     test_prob = model.predict_proba(test.drop(['ID_code'],axis=1))

#     test_prob=test_prob[:,1]

#     test_probs.append(test_prob)

    
# test_probs =[]

# for fold ,(_train_idx ,_val_idx) in enumerate (kfold.split(x_pca,y)):

#     print("####################################")

#     print("############fold:",fold)

#     x_train = x_pca.iloc[_train_idx]

#     y_train = y.iloc[_train_idx]

#     x_val = x_pca.iloc[_val_idx]

#     y_val = y.iloc[_val_idx]

#     f_time = time.time()

#     model.fit(x_train,y_train)

#     print('cost time %s s'%(time.time()-f_time))

#     print('train_roc:%s'%(roc_auc_score(y_train,model.predict_proba(x_train)[:,1])))

#     print('val_roc:%s'%(roc_auc_score(y_val,model.predict_proba(x_val)[:,1])))

#     test_prob = model.predict_proba(test.drop(['ID_code'],axis=1))

#     test_prob=test_prob[:,1]

#     test_probs.append(test_prob)

    
# test_prob = pd.DataFrame(test_probs).mean(axis=0)
# test_prob
submission['target']=test_pred_[:,1]
submission.to_csv("submission.csv",index=False)
submission